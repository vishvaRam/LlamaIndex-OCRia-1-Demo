import asyncio
import base64
import io
import os
import re
import time
import uuid
from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# --- Configuration ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://213.192.2.121:40080/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "Vishva007/OCRia-1-W4A16-GPTQ")
PROMPT_PATH = Path(os.getenv("PROMPT_PATH", "prompt.txt"))
MAX_PIXELS = 4_100_000  # ~210 DPI letter page budget from OCRia-1 benchmarks
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "300"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

if not PROMPT_PATH.exists():
    raise FileNotFoundError(f"System prompt file not found at: {PROMPT_PATH.resolve()}")

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# UI-visible prompt modes. OCRia-1 uses one Markdown transcription prompt;
# all modes map to it so the model's required output format stays intact.
PROMPT_MODES = (
    "prompt_layout_all_en",
    "prompt_layout_only_en",
    "prompt_ocr",
    "prompt_scene_spotting",
    "prompt_markdown",
)

llm_lock = asyncio.Lock()
llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    model=MODEL_NAME,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
settings = {
    "base_url": VLLM_BASE_URL,
    "model": MODEL_NAME,
    "config": {
        "dpi": PDF_RENDER_DPI,
        "num_thread": MAX_CONCURRENCY,
        "max_completion_tokens": None,
        "temperature": 0.0,
    },
}

app = FastAPI(title="OCRia-1 Document Parser API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Pydantic Models ---
class PageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    page_number: int = Field(..., description="1-based page index.")
    markdown: str = Field(
        ...,
        description="Extracted page content in Markdown with HTML tables, stripped of outer code fences.",
    )
    width: int = Field(..., description="Processed image width in pixels.")
    height: int = Field(..., description="Processed image height in pixels.")
    prompt_tokens: int | None = Field(default=None, description="Input tokens consumed for this page.")
    completion_tokens: int | None = Field(default=None, description="Output tokens generated for this page.")


class DocumentParseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    filename: str = Field(..., description="Uploaded file name.")
    total_pages: int = Field(..., description="Total number of pages processed.")
    full_markdown: str = Field(
        ...,
        description="All pages concatenated in reading order separated by page breaks.",
    )
    pages: list[PageResult] = Field(..., description="Per-page structured parsing results.")


class UiPage(BaseModel):
    page: int
    image: str
    width: int
    height: int
    raw: str
    markdown: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class UiRun(BaseModel):
    name: str
    title: str
    mode: str
    prompt_mode: str
    page_count: int
    modified: float
    pages: list[UiPage]


class UiJob(BaseModel):
    id: str
    name: str
    status: str
    done: int
    total: int
    error: str | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")

    base_url: str | None = None
    model: str | None = None
    config: dict[str, object] | None = None


# --- In-memory UI state ---
ui_runs: dict[str, UiRun] = {}
ui_jobs: dict[str, UiJob] = {}


def get_llm() -> ChatOpenAI:
    """Builds the client from current settings so /api/settings changes take effect."""
    return ChatOpenAI(
        base_url=settings["base_url"],
        api_key=VLLM_API_KEY,
        model=settings["model"],
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def probe_vllm() -> dict:
    try:
        import requests

        r = requests.get(settings["base_url"].rstrip("/") + "/models", timeout=5)
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
        return {"reachable": True, "models": models, "error": None}
    except Exception as exc:
        return {"reachable": False, "models": [], "error": str(exc)}


def settings_payload() -> dict:
    return {"serving": "vLLM", "grounding": False, **settings, **probe_vllm()}


# --- Helpers ---
def prepare_image_b64(image_bytes: bytes) -> tuple[str, int, int]:
    """Converts image bytes to RGB, downscales to <= 4.1MP if needed, and encodes as base64 PNG."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        scale = min(1.0, (MAX_PIXELS / (img.width * img.height)) ** 0.5)
        if scale < 1.0:
            new_size = (round(img.width * scale), round(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_str, img.width, img.height


def render_pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = PDF_RENDER_DPI,
    pages: str | None = None,
) -> list[tuple[str, int, int]]:
    """Renders each selected page of a PDF at the target DPI and normalizes it to the 4.1MP budget."""
    wanted = parse_pages(pages)
    rendered_pages: list[tuple[str, int, int]] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if len(doc) == 0:
            raise ValueError("PDF contains zero pages.")
        indices = wanted if wanted is not None else range(len(doc))
        for index in indices:
            if index >= len(doc):
                raise ValueError(f"Page {index + 1} is out of range; PDF has {len(doc)} pages.")
            pix = doc[index].get_pixmap(dpi=dpi)
            rendered_pages.append(prepare_image_b64(pix.tobytes("png")))
    return rendered_pages


def parse_pages(pages: str | None) -> list[int] | None:
    """Parses 'all' or '1,3-5' into zero-based page indices."""
    text = (pages or "").strip().lower()
    if not text or text == "all":
        return None
    indices: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if not start.strip().isdigit() or not end.strip().isdigit():
                raise ValueError(f"Invalid page range: {part}")
            lo, hi = int(start), int(end)
            if lo < 1 or hi < lo:
                raise ValueError(f"Invalid page range: {part}")
            indices.update(range(lo - 1, hi))
        elif part.isdigit():
            n = int(part)
            if n < 1:
                raise ValueError(f"Invalid page number: {part}")
            indices.add(n - 1)
        else:
            raise ValueError(f"Invalid page selection: {part}")
    if not indices:
        raise ValueError("No pages selected.")
    return sorted(indices)


def is_pdf_upload(filename: str, content_type: str) -> bool:
    return content_type == "application/pdf" or filename.lower().endswith(".pdf")


def is_image_upload(filename: str, content_type: str) -> bool:
    return content_type.startswith("image/") or filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp")
    )


def extract_markdown(raw_output: str) -> str:
    """Strips the outer ```markdown ... ``` wrapper emitted by OCRia-1."""
    match = re.search(r"```markdown\n(.*?)(?:\n```|$)", raw_output, re.DOTALL)
    return match.group(1).strip() if match else raw_output.strip()


def build_messages(b64_image: str) -> list[SystemMessage | HumanMessage]:
    """Builds the single-page system + image prompt sequence expected by OCRia-1."""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                }
            ]
        ),
    ]


async def parse_pages_with_model(
    prepared_pages: list[tuple[str, int, int]],
    max_concurrency: int,
) -> list[PageResult]:
    batch_inputs = [build_messages(b64_img) for b64_img, _, _ in prepared_pages]
    try:
        async with llm_lock:
            client = get_llm()
            ai_messages = await client.abatch(
                batch_inputs,
                config={"max_concurrency": max_concurrency},
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM batch inference failed: {exc}") from exc

    page_results: list[PageResult] = []
    for idx, ((_, width, height), ai_msg) in enumerate(zip(prepared_pages, ai_messages), start=1):
        raw_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        clean_md = extract_markdown(raw_text)
        usage = getattr(ai_msg, "usage_metadata", None) or {}
        page_results.append(
            PageResult(
                page_number=idx,
                markdown=clean_md,
                width=width,
                height=height,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
            )
        )
    return page_results


async def run_ui_job(job_id: str, raw_bytes: bytes, filename: str, pages: str, dpi: int) -> None:
    job = ui_jobs[job_id]
    try:
        content_type = ""
        b64_pages: list[tuple[str, int, int]] = []
        if is_pdf_upload(filename, content_type):
            b64_pages = await asyncio.to_thread(render_pdf_to_images, raw_bytes, dpi, pages)
        elif is_image_upload(filename, content_type):
            b64_pages = [await asyncio.to_thread(prepare_image_b64, raw_bytes)]
        else:
            raise ValueError("Unsupported file type. Upload a PDF or an image.")

        job.total = len(b64_pages)
        results = await parse_pages_with_model(b64_pages, MAX_CONCURRENCY)
        ui_pages = [
            UiPage(
                page=i,
                image=f"data:image/png;base64,{b64}",
                width=width,
                height=height,
                raw=page.markdown,
                markdown=page.markdown,
                prompt_tokens=page.prompt_tokens,
                completion_tokens=page.completion_tokens,
            )
            for i, ((b64, width, height), page) in enumerate(zip(b64_pages, results), start=1)
        ]
        stamp = time.time()
        ui_runs[job.name] = UiRun(
            name=job.name,
            title=filename,
            mode="markdown",
            prompt_mode="prompt_markdown",
            page_count=len(ui_pages),
            modified=stamp,
            pages=ui_pages,
        )
        job.done = job.total
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)


# --- Parse endpoints ---
@app.post("/parse", response_model=DocumentParseResponse)
async def parse_document(
    file: UploadFile = File(...),
    max_concurrency: int = Query(
        default=MAX_CONCURRENCY,
        ge=1,
        le=64,
        description="Maximum parallel requests sent to vLLM via abatch.",
    ),
) -> DocumentParseResponse:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded_file"
    content_type = (file.content_type or "").lower()
    is_pdf = is_pdf_upload(filename, content_type)
    is_image = is_image_upload(filename, content_type)

    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF or an image (PNG, JPEG, WebP, TIFF, BMP).",
        )

    try:
        if is_pdf:
            prepared_pages = await asyncio.to_thread(render_pdf_to_images, raw_bytes, PDF_RENDER_DPI)
        else:
            prepared_pages = [await asyncio.to_thread(prepare_image_b64, raw_bytes)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process input file: {exc}") from exc

    page_results = await parse_pages_with_model(prepared_pages, max_concurrency)
    full_markdown = "\n\n---\n\n".join(page.markdown for page in page_results)

    return DocumentParseResponse(
        filename=filename,
        total_pages=len(page_results),
        full_markdown=full_markdown,
        pages=page_results,
    )


# --- Viewer UI endpoints ---
@app.get("/api/info")
async def api_info() -> dict:
    return {"out": f"OCRia-1 · {settings['model']}", "base_url": settings["base_url"]}


@app.get("/api/settings")
async def get_settings() -> dict:
    return settings_payload()


@app.put("/api/settings")
async def put_settings(update: SettingsUpdate) -> dict:
    if update.base_url:
        settings["base_url"] = update.base_url
    if update.model:
        settings["model"] = update.model
    if update.config:
        for key in ("dpi", "num_thread", "max_completion_tokens", "temperature"):
            if key in update.config:
                settings["config"][key] = update.config[key]
    return settings_payload()


@app.get("/api/runs")
async def list_runs() -> list[UiRun]:
    return sorted(ui_runs.values(), key=lambda r: r.modified, reverse=True)


@app.get("/api/runs/{name}")
async def get_run(name: str) -> UiRun:
    run = ui_runs.get(name)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/api/runs/{name}", status_code=204)
async def delete_run(name: str) -> None:
    if name not in ui_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    del ui_runs[name]


@app.get("/api/jobs")
async def list_jobs() -> list[UiJob]:
    return list(ui_jobs.values())


@app.post("/api/runs")
async def create_run(
    file: UploadFile = File(...),
    name: str = "",
    mode: str = "prompt_markdown",
    pages: str = "",
) -> UiJob:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded_file"
    content_type = (file.content_type or "").lower()
    if not (is_pdf_upload(filename, content_type) or is_image_upload(filename, content_type)):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF or an image.",
        )

    if mode not in PROMPT_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown prompt mode: {mode}")

    run_name = (name or Path(filename).stem).strip() or f"run-{uuid.uuid4().hex[:8]}"
    job_id = uuid.uuid4().hex
    job = UiJob(id=job_id, name=run_name, status="running", done=0, total=0)
    ui_jobs[job_id] = job
    asyncio.create_task(run_ui_job(job_id, raw_bytes, filename, pages, settings["config"]["dpi"] or PDF_RENDER_DPI))
    return job


@app.delete("/api/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    if job_id not in ui_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del ui_jobs[job_id]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9632)
