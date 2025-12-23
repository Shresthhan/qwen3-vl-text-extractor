# Qwen3-VL Text Extractor

A tool to extract text and data from images and PDFs using the Qwen3-VL AI model via a remote GPU server.

## Features

- **Quick text extraction** from images and PDFs
- **Reads PDFs** text directly for better accuracy
- **Handles large files** page by page
- **Extracts specific data** (like names, dates) into JSON
- **Simple interface** for all tasks
- **Remote GPU processing** - no powerful local hardware needed

## How it Works

1.  **UI**: Web interface to upload files.
2.  **Backend**: Processes your requests.
3.  **AI Model**: Sends images to a remote GPU server for analysis.

## Requirements

- **Python 3.8 or newer**
- **Internet Connection** (to reach the AI server)
- **API Token**

## Setup

### 1. Install Python Tools

```bash
pip install -r requirements.txt
```

### 2. Configure API Token

Create a `.env` file in the main folder and add your token:

```env
QWEN_API_TOKEN=your_token_here
```

### 3. Start Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start App

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
├── test_images/         # Test files
├── evaluate_model.py    # Testing script
├── requirements.txt     # List of tools needed
└── README.md
```

## Help

**Connection Errors:**
- Check your internet connection.
- Verify your `QWEN_API_TOKEN` in the `.env` file is correct.
- Ensure the backend server is running.

**Error messages:**
- Check the terminal for details.

## License

MIT