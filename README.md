# Qwen3-VL Text Extractor

A local AI-powered text extraction tool using Qwen3-VL-2B-Instruct model running via llama.cpp in Docker.

## Features

- 🚀 Fast text extraction from images
- 🔒 100% local - no data sent to external APIs
- 🎯 Simple web interface built with Streamlit
- 🐳 Dockerized model serving with llama.cpp
- 📝 Customizable extraction prompts

## Architecture

- **Backend**: FastAPI server (`backend/main.py`)
- **Frontend**: Streamlit web UI (`frontend/app.py`)
- **Model Server**: llama.cpp running Qwen3-VL-2B-Instruct in Docker

## Prerequisites

- Docker Desktop
- Python 3.8+
- ~4GB disk space for model files

## Setup

### 1. Download Model Files

Download these files and place them in a `models/` directory:

- `Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf` - Main model file
- `mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf` - Vision projection file

These files are available from Hugging Face.

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Model Server

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

### 4. Start the Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Frontend

```bash
streamlit run frontend/app.py
```

## Usage

1. Open the Streamlit interface (usually at `http://localhost:8501`)
2. Upload an image containing text
3. (Optional) Customize the extraction prompt
4. Click "Extract Text"
5. Download the extracted text

## Default Prompt

The tool uses this prompt by default for pure text extraction:

```
Output only the exact text from this image, without any explanation or commentary. Just the text content.
```

## License

MIT

## Notes

- Model files are NOT included in this repository due to their size (~4GB total)
- The `.gitignore` file excludes the `models/` directory
- First extraction may take longer as the model loads into memory