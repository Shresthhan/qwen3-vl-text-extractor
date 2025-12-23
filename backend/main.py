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

    student_name: str = Field(..., description="Full legal name of the student exactly as written in the offer letter.")

    course_name: Optional[str] = Field(None, description="Exact name of the course/program the student is enrolled in (e.g. 'Bachelor of Information Technology').")

    university_name: Optional[str] = Field(None, description="Official name of the university or college issuing the offer letter (e.g. 'Deakin University').")

    university_address: Optional[str] = Field(None, description="Full postal address of the university/college, including street, city, state/region, and country.")

    total_tuition_amount: Optional[int] = Field(None, description="Total tuition fee for the whole course or stated study period (numeric value only, no currency symbol or commas).")

    remit_amount: Optional[int] = Field(None, description="Initial remittance amount to be paid now (numeric value only, no currency symbol, not the total tuition fee).")

    remit_currency: Optional[str] = Field(None, description="Currency for the remittance amount, e.g. 'AUD', 'USD', 'EUR', 'JPY', 'NRP', or '$'.")

    iban_number: Optional[str] = Field(None, description="IBAN of the beneficiary bank account, including country code and all digits/letters, exactly as printed.")

    swift_code: Optional[str] = Field(None, description="SWIFT/BIC code of the beneficiary bank, exactly as printed (e.g. 'DEUTDEFFXXX').")

    bsb: Optional[str] = Field(None, description="BSB code for Australian bank transfers (if provided)")

    account_number: Optional[str] = Field(None, description="Bank account number of the beneficiary (if provided)")

    bank_name: Optional[str] = Field(None, description="Official name of the beneficiary bank handling the payment (e.g. 'Commonwealth Bank of Australia').")

    payment_purpose: Optional[str] = Field(None, description="Purpose of the payment as described in the letter (e.g. 'Tuition fee deposit' or 'COE deposit').")

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



def extract_text_from_pdf(content: bytes) -> list[str]:
    """Extracts raw text from a PDF file using PyMuPDF, returning a list of page texts."""
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages_text.append(text)
    return pages_text


def stitch_images(pil_images: list[Image.Image]) -> bytes:
    """Stitches a list of PIL images vertically and returns JPEG bytes."""
    if not pil_images:
        raise ValueError("No images to stitch")

    total_width = max(img.width for img in pil_images)
    total_height = sum(img.height for img in pil_images)
    stitched_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in pil_images:
        x_offset = (total_width - img.width) // 2
        stitched_image.paste(img, (x_offset, y_offset))
        y_offset += img.height

    # Only resize if height exceeds 4000 
    max_height = 4000
    if total_height > max_height:
        ratio = max_height / total_height
        new_width = int(total_width * ratio)
        stitched_image = stitched_image.resize(
            (new_width, max_height), Image.Resampling.LANCZOS
        )

    buf = io.BytesIO()
    stitched_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def get_pdf_images(content: bytes) -> list[Image.Image]:
    """Converts all pages of a PDF to a list of PIL Images."""
    doc = fitz.open(stream=content, filetype="pdf")
    pil_images = []
    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        pil_images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
    return pil_images


@app.post("/extract-offer-letter")
async def extract_offer_letter(file: UploadFile = File(...)):
    try:
        content = await file.read()
        is_pdf = file.filename.lower().endswith(".pdf")

        # Start with all fields = None
        merged_data = {field: None for field in OfferLetterData.__fields__}

        def chunk_list(items, chunk_size):
            for i in range(0, len(items), chunk_size):
                yield items[i:i + chunk_size]

        structured_prompt = """
You are extracting structured data from a STUDENT OFFER LETTER used for tuition payment and bank remittance.

Return ONLY valid JSON, no explanations, no markdown, no code blocks.
Use EXACTLY this JSON structure and key names:

{
  "document_type": "offer_letter",
  "extracted_data": {
    "student_name": "<string>",
    "course_name": "<string or null>",
    "university_name": "<string or null>",
    "university_address": "<string or null>",
    "total_tuition_amount": <integer or null>,
    "remit_amount": <integer or null>,
    "remit_currency": "<string>",
    "iban_number": "<string or null>",
    "swift_code": "<string or null>",
    "bsb": "<string or null>",
    "account_number": "<string or null>",
    "bank_name": "<string or null>",
    "payment_purpose": "<string or null>"
  }
}

Field meanings and rules:
- student_name: Full legal name of the student exactly as written in the offer letter.
- course_name: Exact course/program name (for example 'Bachelor of Information Technology').
- university_name: Official name of the university or college issuing the offer letter.
- university_address: Full postal address of the university/college (street, city, state/region, country).
- total_tuition_amount: Total tuition fee for the whole course or defined study period. Use only digits (no commas, no currency symbols).
- remit_amount: Initial remittance/deposit amount to be paid now (not the total tuition amount). Use only digits (no commas, no currency symbols).
- remit_currency: Currency for the remittance amount, for example 'AUD', 'USD', 'EUR', 'JPY', 'NRP' or a symbol like '$'. Always return a non-null string.
- iban_number: IBAN of the beneficiary bank account, including country code and all characters.
- swift_code: SWIFT/BIC code of the beneficiary bank.
- bsb: BSB code for Australian bank transfers. Look for a 6-digit number often formatted as XXX-XXX (e.g., 062-000). Must return this if found. 
- account_number: Bank account number of the beneficiary. Return only if you see account number in the offer letter.
- bank_name: Official name of the beneficiary bank handling the payment. eg. 'Commonwealth Bank of Australia'
- payment_purpose: Purpose of the payment (for example 'Tuition fee deposit', 'COE deposit', or similar phrase). 

General rules:
- STRICTLY EXTRACT VALUES you see in the document. Do NOT return null if a value is present.
- Look for bank details (BSB, IBAN, SWIFT, Account Number) in headers, footers, and payment instruction sections.
- BSB codes usually look like 'BSB: 123-456' or just '123-456'.
- Copy names, codes, and addresses exactly as written.
- For numeric fields, output plain integers only.
- Do NOT add, remove, or rename any keys in the JSON.
Return ONLY the JSON object.
"""



        # ALWAYS use vision/OCR: convert PDF pages (or single image) to PIL images
        if is_pdf:
            pil_images = get_pdf_images(content)  # one PIL image per page
        else:
            pil_images = [Image.open(io.BytesIO(content)).convert("RGB")]

        # Process images page by page (chunk_size=1) for maximum resolution and OCR accuracy
        chunks = list(chunk_list(pil_images, chunk_size=1))
        print(f"DEBUG: Processing {len(chunks)} image chunks for offer letter")

        for i, chunk in enumerate(chunks, start=1):
            try:
                stitched_bytes = stitch_images(chunk)  # high-res stitched JPEG
                print(f"DEBUG: Calling LLM for image chunk {i}")

                raw_response = call_gpu_router_ocr(
                    structured_prompt,
                    stitched_bytes,
                    f"offer_letter_chunk_{i}.jpg",
                )

                cleaned_response = (
                    raw_response.replace("``````", "")
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                extracted_json = json.loads(cleaned_response)
                if isinstance(extracted_json, str):
                    extracted_json = json.loads(extracted_json)

                data_part = extracted_json.get("extracted_data", {})

                # Merge: only fill fields that are still None or empty
                for key, val in data_part.items():
                    # We accept val if it's not None and not an empty string
                    if val is not None and val != "":
                         # We only update if the current stored value is None or empty
                        current_val = merged_data.get(key)
                        if current_val is None or current_val == "":
                            merged_data[key] = val

            except Exception as e:
                print(f"Error processing image chunk {i}: {e}")
                continue

        # Final validation
        final_structure = {
            "document_type": "offer_letter",
            "extracted_data": merged_data,
        }

        try:
            validated = OfferLetterExtraction(**final_structure)
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated.document_type,
                "extracted_data": validated.extracted_data.dict(),
                "validated": True,
                "raw_json": final_structure,
            }
        except ValidationError as e:
            return {
                "success": False,
                "error": "Pydantic validation failed",
                "validation_errors": e.errors(),
                "raw_json": final_structure,
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
