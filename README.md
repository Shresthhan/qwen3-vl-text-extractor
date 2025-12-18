# Qwen3-VL Text Extractor

A powerful local AI-powered text extraction tool using Qwen3-VL-2B-Instruct model running via llama.cpp in Docker. Extract both **unstructured text** and **structured JSON data** from any document image!

## Features

- **Fast text extraction** from images (invoices, receipts, forms, IDs, etc.)
- **Structured data extraction** with automatic JSON validation using Pydantic
- **100% local** - no data sent to external APIs
- **Two Streamlit interfaces** - one for plain text, one for structured data
- **Dockerized model serving** with llama.cpp
- **Customizable extraction prompts**
- **Automatic validation** for structured outputs
- **Model evaluation script** for testing accuracy

## Architecture

```
┌─────────────────────┐
│  Streamlit UIs      │  ← User Interface
│  - app.py           │     Plain text extraction
│  - structured_app.py│     Structured JSON extraction
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FastAPI Backend    │  ← REST API
│  /extract-text      │     Unstructured endpoint
│  /extract-structured│     Structured endpoint
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  llama.cpp Server   │  ← AI Model (Docker)
│  Qwen3-VL-2B        │     Vision Language Model
└─────────────────────┘
```

## Two Extraction Modes

### 1. Unstructured Text Extraction
- **Endpoint**: `/extract-text`
- **Frontend**: `frontend/app.py`
- **Use Case**: Simple text extraction from images
- **Output**: Plain text string

### 2. Structured Data Extraction
- **Endpoint**: `/extract-structured`
- **Frontend**: `frontend/structured_app.py`
- **Use Case**: Extract invoices, receipts, forms, IDs with structured fields
- **Output**: Validated JSON with Pydantic
- **Features**:
  - Automatic document type detection
  - Confidence scoring
  - JSON schema validation
  - Downloadable JSON output

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

### 3. Start the Model Server (Docker)

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

**For Windows PowerShell:**
```powershell
docker run -d --name qwen-extractor `
  -p 8080:8080 `
  -v ${PWD}/models:/models `
  ghcr.io/ggml-org/llama.cpp:server `
  --host 0.0.0.0 `
  -m /models/Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf `
  --mmproj /models/mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf `
  -c 8192
```

### 4. Start the Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Frontend(s)

**Option A: Unstructured Text Extraction**
```bash
streamlit run frontend/app.py
```
Opens at `http://localhost:8501`

**Option B: Structured Data Extraction**
```bash
streamlit run frontend/structured_app.py
```
Opens at `http://localhost:8502` (or next available port)

**Run Both:**
You can run both frontends simultaneously on different ports!

## Usage

### Unstructured Text Extraction
1. Open `http://localhost:8501`
2. Upload an image
3. (Optional) Customize the extraction prompt
4. Click "Extract Text"
5. Download the result

### Structured Data Extraction
1. Open `http://localhost:8502` (or port shown in terminal)
2. Upload a document (invoice, receipt, form, ID card, etc.)
3. (Optional) Provide custom extraction instructions
4. Click "Extract Structured Data"
5. View validated JSON output
6. Download JSON file

## Model Evaluation

Test the model's accuracy across different document types:

```bash
python evaluate_model.py
```

This will test extraction on:
- English Printed Text
- English Handwritten Text
- Nepali Printed Text
- Nepali Handwritten Text

Test images are in `test_images/` directory.

## API Endpoints

### `POST /extract-text`
Extract unstructured text from image

**Request:**
- `file`: Image file (multipart/form-data)
- `prompt`: Optional custom prompt (default: extract all text)

**Response:**
```json
{
  "success": true,
  "extracted_text": "...",
  "filename": "image.png"
}
```

### `POST /extract-structured`
Extract structured data with Pydantic validation

**Request:**
- `file`: Image file (multipart/form-data)
- `custom_prompt`: Optional custom extraction instructions

**Response:**
```json
{
  "success": true,
  "document_type": "invoice",
  "extracted_data": {
    "invoice_number": "INV-001",
    "date": "2024-12-18",
    "total": 500.00
  },
  "confidence": "high",
  "validated": true,
  "raw_json": {...}
}
```

### `GET /health`
Check API and model server status

## Configuration

### Default Unstructured Prompt
```
Output only the exact text from this image, without any explanation or commentary. Just the text content.
```

### Default Structured Prompt
The structured endpoint automatically asks the model to:
- Detect document type
- Extract relevant fields as JSON
- Provide confidence scoring
- Return only valid JSON (no markdown code blocks)

## Project Structure

```
qwen3-vl-text-extractor/
├── backend/
│   └── main.py              # FastAPI server with both endpoints
├── frontend/
│   ├── app.py               # Streamlit UI for unstructured extraction
│   └── structured_app.py    # Streamlit UI for structured extraction
├── models/                  # Model files (gitignored)
│   ├── Qwen_Qwen3-VL-2B-Instruct-Q4_K_M.gguf
│   └── mmproj-Qwen_Qwen3-VL-2B-Instruct-f16.gguf
├── test_images/             # Sample images for evaluation
│   ├── english_printed.png
│   ├── english_handwritten.png
│   ├── nepali_printed.png
│   └── nepali_handwritten.png
├── evaluate_model.py        # Model evaluation script
├── requirements.txt         # Python dependencies
└── README.md
```

## Use Cases

- **Invoice Processing** - Extract invoice numbers, dates, amounts
- **Receipt Digitization** - Capture merchant, items, total
- **Form Automation** - Parse form fields into JSON
- **ID Card Extraction** - Extract name, DOB, ID number
- **Contract Analysis** - Extract key terms and dates
- **Report Processing** - Convert reports to structured data

## Notes

- Model files are **NOT included** in this repository due to size (~4GB total)
- The `.gitignore` file excludes the `models/` directory
- First extraction may take longer as the model loads into memory
- Structured extraction uses Pydantic for automatic JSON validation
- The model runs entirely locally - no internet required after setup

## Troubleshooting

**Backend Error: "cannot access local variable 'cleaned_response'"**
- This has been fixed in the latest version
- Make sure you're running the updated `backend/main.py`

**Docker container won't start:**
- Check if port 8080 is already in use
- Ensure model files are in the correct directory
- Verify Docker has enough memory allocated (8GB+ recommended)

**Streamlit won't connect to backend:**
- Ensure the FastAPI server is running on port 8000
- Check `http://localhost:8000/health` to verify backend status

## License

MIT

## Credits

- **Model**: [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen) by Alibaba Cloud
- **Inference Engine**: [llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Frameworks**: FastAPI, Streamlit, Pydantic