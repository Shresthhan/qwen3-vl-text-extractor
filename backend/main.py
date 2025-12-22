from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import requests
import base64
from typing import Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import re

app = FastAPI(title="Document Extractor API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class NepalNationalIDData(BaseModel):
    # Front side visible fields on the card you showed
    nationality: str                 # e.g. "Nepalese"
    sex: Optional[str] = None        # e.g. "F" / "M"
    surname_english: str             # e.g. "Koirala Pokhrel"
    given_name_english: str          # e.g. "Bhagawati Kumari"
    surname_local: Optional[str] = None   # e.g. "कोइराला पोखरेल"
    given_name_local: Optional[str] = None # e.g. "भगवती कुमारी"
    ninn: str                        # National Identity Number, e.g. "023-456-2130"
    ninn_secondary: Optional[str] = None  # the second printed line of the number
    date_of_birth: str               # e.g. "1978-02-05"
    date_of_issue: str               # e.g. "01-01-2017"
    mothers_name_english: Optional[str] = None
    fathers_name_english: Optional[str] = None
    mothers_name_local: Optional[str] = None
    fathers_name_local: Optional[str] = None
    signature_present: Optional[bool] = None  # true if signature field is visible
    card_type_label: Optional[str] = None     # e.g. "NATIONAL IDENTITY CARD"


class OfferLetterData(BaseModel):
    candidate_name: str
    position: str
    company_name: str
    start_date: str
    salary: Optional[str] = None
    location: Optional[str] = None

class NepalNationalIDExtraction(BaseModel):
    document_type: str
    extracted_data: NepalNationalIDData
    confidence: Optional[str] = "medium"

class OfferLetterExtraction(BaseModel):
    document_type: str
    extracted_data: OfferLetterData
    confidence: Optional[str] = "medium"

class DynamicExtraction(BaseModel):
    document_type: str
    extracted_data: Dict[str, Any]
    confidence: Optional[str] = "medium"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def prepare_stitched_image(file: UploadFile) -> Tuple[str, str]:
    content = await file.read()
    pil_images = []

    if file.filename.lower().endswith('.pdf'):
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("Empty PDF file")
        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_images.append(Image.open(io.BytesIO(img_data)).convert('RGB'))
    else:
        pil_images.append(Image.open(io.BytesIO(content)).convert('RGB'))

    if not pil_images:
        raise ValueError("No images processed")

    total_width = max(img.width for img in pil_images)
    total_height = sum(img.height for img in pil_images)
    stitched_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in pil_images:
        x_offset = (total_width - img.width) // 2
        stitched_image.paste(img, (x_offset, y_offset))
        y_offset += img.height

    MAX_HEIGHT = 3000
    if total_height > MAX_HEIGHT:
        ratio = MAX_HEIGHT / total_height
        new_width = int(total_width * ratio)
        stitched_image = stitched_image.resize((new_width, MAX_HEIGHT), Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    stitched_image.save(output_buffer, format='JPEG', quality=85)
    final_image_bytes = output_buffer.getvalue()
    image_base64 = base64.b64encode(final_image_bytes).decode('utf-8')
    mime_type = "image/jpeg"
    return image_base64, mime_type

def call_llm_and_clean_json(prompt: str, image_base64: str, mime_type: str, temperature: float = 0.3) -> Tuple[str, str]:
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
        ]}],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    response = requests.post(LLAMA_SERVER_URL, json=payload)
    if response.status_code != 200:
        raise Exception(f"LLM Server Error: {response.text}")
    
    result = response.json()
    raw_response = result['choices'][0]['message']['content']
    
    # Clean markdown
    cleaned_response = re.sub(r'```json|```', '', raw_response).strip()
    cleaned_response = re.sub(r'```\s*', '', cleaned_response).strip()
    
    return raw_response, cleaned_response

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/extract-text-image")
async def extract_text_image(file: UploadFile = File(...), prompt: str = Form("Extract all text")):
    if file.filename.lower().endswith('.pdf'):
        return {"success": False, "error": "Use /extract-text-pdf for PDFs"}
    
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    response = requests.post(LLAMA_SERVER_URL, json=payload)
    result = response.json()
    return {"success": True, "extracted_text": result['choices'][0]['message']['content'], "filename": file.filename}

@app.post("/extract-text-pdf")
async def extract_text_pdf(file: UploadFile = File(...), prompt: str = Form("Extract all text")):
    if not file.filename.lower().endswith('.pdf'):
        return {"success": False, "error": "Use /extract-text-image for images"}
    
    image_base64, mime_type = await prepare_stitched_image(file)
    
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
        ]}],
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    response = requests.post(LLAMA_SERVER_URL, json=payload)
    result = response.json()
    return {"success": True, "extracted_text": result['choices'][0]['message']['content'], "filename": file.filename}

@app.post("/extract-national-id")
async def extract_national_id(file: UploadFile = File(...)):
    try:
        image_base64, mime_type = await prepare_stitched_image(file)

        prompt = """
You are extracting data from a NEPAL NATIONAL IDENTITY CARD.

Return ONLY valid JSON, no explanations, no markdown, no code blocks.
Use EXACTLY this JSON structure:

{
  "document_type": "nepal_national_id",
  "extracted_data": {
    "nationality": "<string like 'Nepalese'>",
    "sex": "<'M' or 'F' or null>",
    "surname_english": "<surname in English exactly as printed>",
    "given_name_english": "<given name in English exactly as printed>",
    "surname_local": "<surname in Nepali script or null>",
    "given_name_local": "<given name in Nepali script or null>",
    "ninn": "<primary national identity number as printed>",
    "ninn_secondary": "<secondary line of the number or null>",
    "date_of_birth": "<YYYY-MM-DD as printed>",
    "date_of_issue": "<DD-MM-YYYY or as printed>",
    "mothers_name_english": "<mother's name in English or null>",
    "fathers_name_english": "<father's name in English or null>",
    "mothers_name_local": "<mother's name in Nepali script or null>",
    "fathers_name_local": "<father's name in Nepali script or null>",
    "signature_present": "<true if a signature field is visible, else false>",
    "card_type_label": "<text like 'NATIONAL IDENTITY CARD' or null>"
  },
  "confidence": "<high|medium|low>"
}

Rules:
- Read all English and Nepali text on the card.
- Copy names and numbers exactly as printed (including dashes in NIN).
- If any field is not visible, set it to null (except boolean which should be true/false).
- Do NOT add any extra keys.
Return ONLY the JSON object.
"""

        raw_response, cleaned_response = call_llm_and_clean_json(
            prompt, image_base64, mime_type
        )

        try:
            extracted_json = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": "Invalid JSON from model",
                "raw_response": raw_response,
                "validation_error": str(e),
            }

        try:
            validated = NepalNationalIDExtraction(**extracted_json)
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated.document_type,
                "extracted_data": validated.extracted_data.dict(),
                "confidence": validated.confidence,
                "validated": True,
                "raw_json": extracted_json,
            }
        except ValidationError as e:
            return {
                "success": False,
                "error": "Pydantic validation failed",
                "validation_errors": e.errors(),
                "raw_json": extracted_json,
            }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/extract-offer-letter")
async def extract_offer_letter(file: UploadFile = File(...)):
    try:
        image_base64, mime_type = await prepare_stitched_image(file)
        
        prompt = """You are extracting OFFER LETTER data. Return ONLY this exact JSON:
{
  "document_type": "offer_letter",
  "extracted_data": {
    "candidate_name": "",
    "position": "",
    "company_name": "",
    "start_date": "",
    "salary": null,
    "location": null
  },
  "confidence": "medium"
}"""
        
        raw_response, cleaned_response = call_llm_and_clean_json(prompt, image_base64, mime_type)
        extracted_json = json.loads(cleaned_response)
        validated = OfferLetterExtraction(**extracted_json)
        
        return {
            "success": True,
            "filename": file.filename,
            "document_type": validated.document_type,
            "extracted_data": validated.extracted_data.dict(),
            "confidence": validated.confidence,
            "validated": True,
            "raw_json": extracted_json
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    try:
        requests.get("http://localhost:8080/health")
        return {"status": "healthy"}
    except:
        return {"status": "degraded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
