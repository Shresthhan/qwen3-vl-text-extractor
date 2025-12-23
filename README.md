# Qwen3-VL Text Extractor

A local tool to extract text and data from images and PDFs using the Qwen3-VL AI model.

## Features

- **Quick text extraction** from images and PDFs
- **Reads PDFs** text directly for better accuracy
- **Handles large files** page by page
- **Extracts specific data** (like names, dates) into JSON
- **100% offline** - keeps your data private
- **Simple interface** for all tasks
- **Uses Docker** to run the AI model

## How it Works

1.  **UI**: Web interface to upload files.
2.  **Backend**: Processes your requests.
3.  **AI Model**: Reads the images and text.

## Requirements

- **Docker Desktop**
- **Python 3.8 or newer**
- **4GB disk space**
- **8GB RAM** (recommended)

## Setup

### 1. Get Model Files

Download these two files to a `models/` folder:
- `Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf`

### 2. Install Python Tools

```bash
pip install -r requirements.txt
```

### 3. Settings (Optional)

Create a `.env` file if you have an API token:

```env
QWEN_API_TOKEN=your_token
```

### 4. Start AI Server via Docker

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

### 5. Start Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start App

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`.

## How to Use

### Get Text
1. Open the app.
2. Choose "Unstructured text".
3. Upload file.
4. Click "Extract Text".

### Get Data (JSON)
1. Choose "Structured JSON".
2. Pick a type (like National ID).
3. Upload file.
4. Click "Extract Structured Data".
5. Download the result.

## Files

```
qwen3-vl-text-extractor/
├── backend/             # Server code
├── frontend/            # App interface
├── models/              # AI model files
├── test_images/         # Test files
├── evaluate_model.py    # Testing script
├── requirements.txt     # List of tools needed
└── README.md
```

## Help

**Docker won't start:**
- Check if port 8080 is free.
- Check if model files are in `models/`.
- Check if Docker has enough memory.

**Error messages:**
- Check the terminal for details.

## License

MIT