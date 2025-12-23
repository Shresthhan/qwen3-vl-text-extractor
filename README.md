# Qwen3-VL Text Extractor

A powerful local AI-powered text extraction tool using Qwen3-VL-2B-Instruct model running via llama.cpp in Docker. Extract both unstructured text and structured JSON data from any document image or PDF.

## Features

- **Fast text extraction** from images and PDFs
- **Hybrid PDF Processing** that extracts raw text directly from digital PDFs for perfect accuracy, falling back to vision only for scanned documents
- **Iterative Chunking** for large PDFs to handle unlimited pages without hitting token limits
- **Structured data extraction** with automatic JSON validation using Pydantic
- **100% local** - no data sent to external APIs
- **Unified Streamlit Interface** for both text and structured data
- **Dockerized model serving** with llama.cpp
- **Customizable extraction prompts**
- **Automatic validation** for structured outputs

## Architecture

    ┌─────────────────────┐
    │  Streamlit UI       │  ← User Interface
    │  - app.py           │     Combined extraction interface
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  FastAPI Backend    │  ← REST API
    │  /extract-text-pdf  │     PDF text extraction
    │  /extract-text-image│     Image text extraction
    │  /extract-national-id     Structured ID extraction
    │  /extract-offer-letter    Structured Offer Letter extraction
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  llama.cpp Server   │  ← AI Model (Docker)
    │  Qwen3-VL-2B        │     Vision Language Model
    └─────────────────────┘

## Prerequisites

- **Docker Desktop** (for running the model server)
- **Python 3.8+**
- **~4GB disk space** for model files
- **8GB+ RAM** recommended

## Setup

### 1. Download Model Files

Download these files and place them in a `models/` directory:

- `Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf` - Main model file
- `mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf` - Vision projection file

> Available from [Hugging Face](https://huggingface.co)

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```env
QWEN_API_TOKEN=your_token_here (optional, if you have set up auth)
```

### 4. Start the Model Server (Docker)

```bash
docker run -d --name qwen-extractor \
  -p 8080:8080 \
  -v ${PWD}/models:/models \
  ghcr.io/ggml-org/llama.cpp:server \
  --host 0.0.0.0 \
  -m /models/Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj /models/mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf \
  -c 8192
```

### 5. Start the Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Frontend

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`

## Usage

### Unstructured Text Extraction
1. Open the UI
2. Select "Unstructured text" mode
3. Upload an image or PDF
4. Click "Extract Text"

### Structured Data Extraction
1. Select "Structured JSON" mode
2. Choose "National ID" or "Offer Letter"
3. Upload the document
4. Click "Extract Structured Data"
5. Download the validated JSON

## Project Structure

```
qwen3-vl-text-extractor/
├── backend/
│   └── main.py              # FastAPI server
├── frontend/
│   └── app.py               # Streamlit UI
├── models/                  # Model files (gitignored)
├── test_images/             # Sample images
├── evaluate_model.py        # Model evaluation script
├── requirements.txt         # Python dependencies
└── README.md
```

## Troubleshooting

**Docker container won't start:**
- Check if port 8080 is already in use
- Ensure model files are in the correct directory
- Verify Docker has enough memory allocated (8GB+ recommended)

**Backend Error:**
- Check the console logs for detailed error messages
- Ensure `python-dotenv` is installed if using `.env`

## License

MIT