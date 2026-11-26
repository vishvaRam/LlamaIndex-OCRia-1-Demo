# OCRia-1 Document Parser

FastAPI app for converting PDFs and images to Markdown with OCRia-1 served through vLLM.

## Run

```bash
uv sync
uv run uvicorn api:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` for the browser viewer.

## Use

### Browser viewer

1. Click **New run**.
2. Choose a PDF or image.
3. Click **Run OCR**.
4. View per-page Markdown beside the source image.

Use the hamburger button to hide/show the sidebar. Use **Raw** to see the untouched model output.

### API endpoint

```bash
curl -X POST "http://localhost:8080/parse" \
  -F "file=@document.pdf"
```

Response fields:

- `full_markdown`: all pages joined in reading order
- `pages`: per-page Markdown, dimensions, and token usage

## Configuration

Set these before starting the server:

```bash
VLLM_BASE_URL=http://213.192.2.122:40135/v1
VLLM_API_KEY=EMPTY
MODEL_NAME=llamaindex/OCRia-1
PROMPT_PATH=prompt.txt
PDF_RENDER_DPI=300
MAX_CONCURRENCY=8
```

The vLLM base URL and model can also be changed from the viewer's model-server settings panel.
