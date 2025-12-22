from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, Field
import requests
import base64
from typing import Dict, Any, Optional, Tuple
import fitz
from PIL import Image
import io
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Document Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = "https://gpu-router.server247.info/route/qwen3-vl"
API_TOKEN = os.getenv("QWEN_API_TOKEN")


class NepalNationalIDCardData(BaseModel):
    nationality: str                  
    sex: Optional[str] = None         
    surname_english: str              
    given_name_english: str           
    
    surname_nepali: Optional[str] = None   
    given_name_nepali: Optional[str] = None
    
    ninn_nepali: str                  
    ninn_english: Optional[str] = None  
    
    date_of_birth_ad: str 
    date_of_birth_bs: str               
    date_of_issue: str               
    
    mothers_name: Optional[str] = None
    fathers_name: Optional[str] = None


class OfferLetterData(BaseModel):

    course_name: str = Field(..., description="Name of the course/program the student is admitted to")
    student_name: str = Field(..., description="Full name of the student")

    total_tuition_amount: float = Field(..., description="Total tuition amount (numeric only, no currency symbol)")
    total_tuition_currency: str = Field(..., description="Currency of the total tuition amount, e.g. 'AUD', 'USD', 'EUR'")
    remit_amount: float = Field(..., description="Amount to be remitted/paid now (numeric only, no currency symbol)")
    remit_currency: str = Field(..., description="Currency of the remit amount, e.g. 'AUD', 'USD', 'EUR'")

    beneficiary_name: str = Field(..., description="Name of the university or college (beneficiary)")
    university_address: str = Field(..., description="Full postal address of the university/beneficiary")

    iban: Optional[str] = Field(None, description="IBAN code for the payment (if provided)")
    swift: Optional[str] = Field(None, description="SWIFT/BIC code for the payment (if provided)")
    bsb: Optional[str] = Field(None, description="BSB code for Australian bank transfers (if provided)")
    account_number: Optional[str] = Field(None, description="Bank account number of the beneficiary (if provided)")
    bank_name: Optional[str] = Field(None, description="Name of the bank (if explicitly mentioned)")

    payment_purpose: Optional[str] = Field(None, description="Purpose/reference for the payment of remit amount")
    payment_reference: Optional[str] = Field(None, description="Specific payment reference or student ID to use in bank transfer")


class NepalNationalIDCardExtraction(BaseModel):
    document_type: str               
    extracted_data: NepalNationalIDCardData



class OfferLetterExtraction(BaseModel):
    document_type: str
    extracted_data: OfferLetterData


class DynamicExtraction(BaseModel):
    document_type: str
    extracted_data: Dict[str, Any]


async def prepare_stitched_image(file: UploadFile) -> Tuple[bytes, str]:
    content = await file.read()
    pil_images = []

    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("Empty PDF file")
        for page in doc:
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
    else:
        pil_images.append(Image.open(io.BytesIO(content)).convert("RGB"))

    if not pil_images:
        raise ValueError("No images processed")

    total_width = max(img.width for img in pil_images)
    total_height = sum(img.height for img in pil_images)
    stitched_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in pil_images:
        x_offset = (total_width - img.width) // 2
        stitched_image.paste(img, (x_offset, y_offset))
        y_offset += img.height

    max_height = 3000
    if total_height > max_height:
        ratio = max_height / total_height
        new_width = int(total_width * ratio)
        stitched_image = stitched_image.resize(
            (new_width, max_height), Image.Resampling.LANCZOS
        )

    buf = io.BytesIO()
    stitched_image.save(buf, format="JPEG", quality=85)
    final_bytes = buf.getvalue()
    return final_bytes, "image/jpeg"


def call_gpu_router_ocr(prompt: str, image_bytes: bytes, filename: str) -> str:
    files = {
        "image": (filename, image_bytes, "image/jpeg"),
    }
    data = {
        "system_prompt": "You are an OCR/information extraction assistant.",
        "prompt": prompt,
    }

    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    resp = requests.post(LLAMA_SERVER_URL, data=data, files=files, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"LLM Server Error: {resp.status_code} - {resp.text}")

    return resp.text.strip()


@app.post("/extract-text-image")
async def extract_text_image(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this image"),
):
    if file.filename.lower().endswith(".pdf"):
        return {"success": False, "error": "Use /extract-text-pdf for PDF files"}

    image_bytes = await file.read()

    try:
        extracted_text = call_gpu_router_ocr(prompt, image_bytes, file.filename)
        return {
            "success": True,
            "extracted_text": extracted_text,
            "filename": file.filename,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/extract-text-pdf")
async def extract_text_pdf(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this PDF document"),
):
    if not file.filename.lower().endswith(".pdf"):
        return {"success": False, "error": "Use /extract-text-image for image files"}

    try:
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) == 0:
            return {"success": False, "error": "Empty PDF file"}

        all_extracted_text = []

        for i, page in enumerate(doc):
            # Render high-res image for the page
            # Matrix(2,2) = 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("jpeg")
            
            # Extract text for this specific page
            page_text = call_gpu_router_ocr(prompt, img_bytes, f"page_{i+1}.jpg")
            all_extracted_text.append(f"--- Page {i+1} ---\n{page_text}")
        
        final_text = "\n\n".join(all_extracted_text)

        return {
            "success": True,
            "extracted_text": final_text,
            "filename": file.filename,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/extract-national-id")
async def extract_national_id(file: UploadFile = File(...)):
    try:
        stitched_bytes, mime_type = await prepare_stitched_image(file)

        structured_prompt = """
You are extracting data from a NEPAL NATIONAL IDENTITY CARD.

Return ONLY valid JSON, no explanations, no markdown, no code blocks.
Use EXACTLY this JSON structure:

{
  "document_type": "nepal_national_id",
  "extracted_data": {
    "nationality": "<string like 'Nepalese'>",
    "sex": "<'M' or 'F' or null>",
    "surname_english": "<surname in English (include ALL words you detect)>",
    "given_name_english": "<given name in English (include ALL words you detect)>",
    "surname_nepali": "<surname in Nepali script (include ALL words you detect)>",
    "given_name_nepali": "<given name in Nepali script (include ALL words you detect)>",
    "ninn_nepali": "<first line of national identity number, exactly as printed in Nepali numbers>",
    "ninn_english": "<second line of national identity number in english>",
    "date_of_birth_ad": "<AD format>",
    "date_of_birth_bs": "convert AD to BS",
    "date_of_issue": "<YYYY-MM-DD format>",
    "mothers_name": "<mother's name in Nepali script>",
    "fathers_name": "<father's name in Nepali script>"
  }
}

Rules:
- Read ALL English and Nepali text on the card.
- Read date of birth in AD and convert to BS.
- Print nepali text as it is.
- Copy names and numbers exactly as printed (including dashes in NIN).
- Do NOT add any extra keys.
Return ONLY the JSON object.
"""

        raw_response = call_gpu_router_ocr(
            structured_prompt, stitched_bytes, "national_id.jpg"
        )

        # Clean possible markdown wrappers
        cleaned_response = raw_response.strip()
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()

        try:
            extracted_json = json.loads(cleaned_response)
            # Handle double-encoded JSON (string that itself is JSON)
            if isinstance(extracted_json, str):
                try:
                    extracted_json = json.loads(extracted_json)
                except Exception:
                    pass
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": "Invalid JSON from model",
                "raw_response": raw_response,
                "validation_error": str(e),
            }

        try:
            validated = NepalNationalIDCardExtraction(**extracted_json)
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated.document_type,
                "extracted_data": validated.extracted_data.dict(),
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
        stitched_bytes, mime_type = await prepare_stitched_image(file)

        structured_prompt = """
You are extracting structured data from an OFFER LETTER.

Return ONLY valid JSON, no explanations, no markdown, no code blocks.
Use EXACTLY this JSON structure:

{
  "document_type": "offer_letter",
  "extracted_data": {
    "course_name": "<string>",
    "student_name": "<string>",
    "total_tuition_amount": <number>,
    "total_tuition_currency": "<string like 'AUD' or 'USD'>",
    "remit_amount": <number>,
    "remit_currency": "<string like 'AUD' or 'USD'>",
    "beneficiary_name": "<string>",
    "university_address": "<string>",
    "iban": "<string or null>",
    "swift": "<string or null>",
    "bsb": "<string or null>",
    "account_number": "<string or null>",
    "bank_name": "<string or null>",
    "payment_purpose": "<string or null>",
    "payment_reference": "<string or null>"
  }
}

Rules:
- Read ALL pages of the offer letter.
- Copy names, amounts, currencies, bank details, and addresses exactly as printed.
- For amounts, use only numbers (no currency symbols); currencies go in the currency fields.
- If any field is not present in the document, set its value to null.
- Do NOT add, remove, or rename any keys.
Return ONLY the JSON object.
"""


        raw_response = call_gpu_router_ocr(
            structured_prompt, stitched_bytes, "offer_letter.jpg"
        )

        cleaned_response = (
            raw_response.replace("``````", "").strip()
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
            validated = OfferLetterExtraction(**extracted_json)
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated.document_type,
                "extracted_data": validated.extracted_data.dict(),
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


@app.get("/health")
async def health_check():
    try:
        resp = requests.post(
            LLAMA_SERVER_URL,
            data={"system_prompt": "ping", "prompt": "ping"},
        )
        if resp.status_code == 200:
            return {"status": "healthy", "router": "online"}
        return {"status": "degraded", "router_status_code": resp.status_code}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
