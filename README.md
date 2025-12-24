# Qwen3-VL Text Extractor

A tool to extract text and data from images and PDFs using the Qwen3-VL AI model via a remote GPU server.

## Features

- **Quick text extraction** from images and PDFs
- **Reads PDFs** text directly for better accuracy
- **Handles large files** page by page
- **Extracts specific data** (like names, dates) into JSON
- **Automatic transliteration** - converts non-Latin scripts (Japanese, Chinese, Arabic, etc.) to English alphabet
- **Smart field merging** - hybrid system with code rules and LLM judge for handling multi-page documents
- **Simple interface** for all tasks
- **Remote GPU processing** - no powerful local hardware needed

## Program Flow & Logic

### 1. Data Extraction Flow

The system processes documents through multiple specialized layers to ensure maximum accuracy:

- **Preprocessing**: Large PDFs are split into single-page chunks. Each page is converted to a high-resolution JPEG to optimize visibility for the vision model.
- **Vision Extraction**: Each page is analyzed by the **Qwen3-VL-8B** model. It identifies key fields and returns them in a structured JSON format.

### 2. Hybrid Merging System

For multi-page documents, the system needs to combine information from different pages. We use a **Hybrid Merge** approach:

- **Step 1: Code Rules (Fast Path)**: Deterministic Python rules handle common merging scenarios:
  - Preferring 3-letter currency codes (e.g., "AUD" over "$").
  - Script Preference: Favoring Latin script (English) names for universities and banks over native script or transliterated versions.
  - Validating numeric fields (amounts, account numbers).
  - Preferring longer, more complete addresses.
  - Handling payment purposes (preferring specific purposes over generic ones).
- **Step 2: LLM Judge (Slow Path)**: If code rules cannot decide between two conflicting values, a dedicated LLM call acts as a "judge" to select the most contextually relevant value.

### 3. Post-Processing

After the data is merged, it passes through the final processing layer:

- **Currency Normalization**: Common symbols (e.g., "$", "€", "£") and names (e.g., "Dollars", "Euro") are automatically converted to their standard ISO 4217 3-letter codes (e.g., "USD", "EUR", "GBP").
- **Automatic Romanization**: The system detects non-Latin scripts (Chinese, Arabic, Japanese, Korean, Thai, Cyrillic, etc.) and uses the **Transliteration Engine** to convert them into the Latin alphabet phonetically.

### 4. Final Validation

All extracted and processed data is validated against a strict **Pydantic schema** to ensure the final JSON is consistent and error-free.

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
