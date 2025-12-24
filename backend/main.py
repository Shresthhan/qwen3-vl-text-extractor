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


# ========================================
# PYDANTIC MODELS
# ========================================

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


# ========================================
# UTILITY FUNCTIONS
# ========================================

async def prepare_stitched_image(file: UploadFile) -> Tuple[bytes, str]:
    """Prepares a stitched image from uploaded file (PDF or image)."""
    content = await file.read()
    pil_images = []

    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        if len(doc) == 0:
            raise ValueError("Empty PDF file")
        for page in doc:
            mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better quality
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
    """Calls the GPU router OCR API."""
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
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        pil_images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
    return pil_images


# ========================================
# HYBRID FIELD MERGE SYSTEM
# ========================================

def clean_student_name(name: str) -> str:
    """
    Post-process student name to enforce proper capitalization.
    This ensures consistent formatting regardless of LLM output.
    """
    if not name or not isinstance(name, str):
        return name
    
    # Remove common unwanted patterns
    unwanted_patterns = [
        "student id:", "id:", "sid:", "dob:", "date of birth:", 
        "email:", "@", "mr.", "ms.", "mrs.", "dr.", "prof.", 
        "professor", "sir", "madam", "miss"
    ]
    
    cleaned = name.strip()
    cleaned_lower = cleaned.lower()
    
    # Remove unwanted text
    for pattern in unwanted_patterns:
        if pattern in cleaned_lower:
            # Find the pattern and remove everything from that point
            idx = cleaned_lower.find(pattern)
            cleaned = cleaned[:idx].strip()
            cleaned_lower = cleaned.lower()
    
    # Remove any remaining numbers
    cleaned = ''.join(char for char in cleaned if not char.isdigit())
    
    # Apply proper title case: First letter of each word capitalized
    # This handles "SAKIN POUDEL", "sakin poudel", "Sakin POUDEL" all correctly
    words = cleaned.split()
    proper_cased_words = [word.capitalize() for word in words if word.strip()]
    
    result = ' '.join(proper_cased_words)
    return result.strip()


def transliterate_to_latin(text: str, field_name: str = "") -> str:
    """
    Transliterate non-Latin scripts to Latin alphabet using LLM.
    Returns the original text if it's already in Latin script or if transliteration fails.
    """
    if not text or not isinstance(text, str):
        return text
    
    # Quick check: if text is mostly ASCII/Latin, skip transliteration
    latin_chars = sum(1 for c in text if ord(c) < 128 or (ord(c) >= 192 and ord(c) <= 591))
    total_chars = len([c for c in text if c.strip()])
    
    if total_chars == 0:
        return text
    
    # If more than 80% is already Latin script, skip
    if latin_chars / total_chars > 0.8:
        return text
    
    # Use LLM to transliterate
    transliteration_prompt = f"""You are a transliteration expert. Your job is to convert non-Latin scripts to Latin alphabet (romanization).

Input text: "{text}"
Field type: {field_name}

CRITICAL RULES:
- TRANSLITERATE (convert script/alphabet), DO NOT TRANSLATE (convert meaning)
- Convert characters to their phonetic Latin equivalent
- Keep the same meaning/pronunciation, just change the writing system
- Examples:
  * Japanese "東京福祉大学" → "Tokyo Fukushi Daigaku"
  * Chinese "北京大学" → "Beijing Daxue"
  * Arabic "جامعة القاهرة" → "Jamiat al-Qahira"
  * Korean "서울대학교" → "Seoul Daehakgyo"
  * Thai "มหาวิทยาลัย" → "Mahawitthayalai"

Return ONLY the romanized text, nothing else. No explanations, no quotes, just the transliterated text.
"""
    
    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            data={
                "system_prompt": "You are a transliteration expert. Return only the romanized text.",
                "prompt": transliteration_prompt
            },
            headers={"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.text.strip()
            # Remove quotes if LLM added them
            result = result.strip('"').strip("'").strip()
            
            if result and len(result) > 0:
                # print(f"   🌐 TRANSLITERATE: '{text}' → '{result}'")
                return result
        
        # If LLM fails, return original
        # print(f"   ⚠️ Transliteration failed for '{text}', keeping original")
        return text
        
    except Exception as e:
        # print(f"   ❌ Transliteration error: {e}, keeping original")
        return text


def llm_judge(field_name: str, val_a: Any, val_b: Any) -> Any:
    """
    Call LLM to judge between two conflicting values.
    Only used for ambiguous cases where code rules can't decide.
    """
    judge_prompt = f"""You are a data quality judge for student offer letter extraction.

Field: {field_name}
Value A: "{val_a}"
Value B: "{val_b}"

Choose the BETTER value based on these rules:

student_name:
- Your job is to return a clean, well-formatted full name with 2–4 words.
- Start from the raw extracted value for student_name.
- Remove any IDs, labels, or extra text such as: "ID:", "Student ID", "SID", "DOB:", "Date of Birth", "Email", "@", phone numbers, or any numbers.
- Remove any academic or title words such as: "Mr", "Ms", "Mrs", "Dr", "Prof", "Professor", "Sir", "Madam", "Miss".
- The final result must contain ONLY the person's name.
- Convert the name to proper capitalization:
  - All letters should be lowercase except the first letter of each word.
  - Example: "SAKIN POUDEL" or "sakin poudel" → "Sakin Poudel".
  - Example: "JOHN MICHAEL DOE" → "John Michael Doe".
- Do not return quotation marks around the name.
- Do not add any extra words that were not in the original name.
- If the extracted text is "John Doe Student ID: 12345", the correct cleaned value is "John Doe".

university_name / course_name:
- Prefer official clean name
- Reject if includes campus location or extra details
- Example: "Deakin University" is better than "Deakin University Melbourne Campus"

swift_code / bsb / iban_number / account_number:
- Prefer exact code format without explanations
- Example: "ABCDAU2S" is better than "SWIFT: ABCDAU2S (Commonwealth Bank)"

total_tuition_amount / remit_amount:
- Prefer clean integer
- Reject if contains symbols, commas, text
- Example: "50000" is better than "50,000 AUD"

remit_currency:
- Prefer 3-letter ISO code over symbols
- Example: "AUD" is better than "$"

bank_name:
- Prefer official full name
- Example: "Commonwealth Bank of Australia" is better than "CommBank"

payment_purpose:
- Prefer MORE SPECIFIC payment purpose over generic ones
- If one says "living expenses" and other says "tuition fee", choose the one that matches the document context
- If one is more detailed (e.g., "Living expenses for accommodation"), prefer it over generic (e.g., "tuition fee")
- Do NOT default to "tuition fee" - extract what's actually stated
- Examples: "Living expenses" is valid, "Accommodation fee" is valid, "Application fee" is valid

show both option A and B in the response
Reply with ONLY one character: "A" or "B"
No explanations, no punctuation, just the letter.
"""

    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            data={
                "system_prompt": "You are a data quality judge. Reply ONLY with A or B.",
                "prompt": judge_prompt
            },
            headers={"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {},
            timeout=5
        )
        
        decision = response.text.strip().upper()
        
        # Extract first letter that is A or B
        for char in decision:
            if char == 'A':
                print(f"      🤖 LLM chose A: '{val_a}'")
                return val_a
            elif char == 'B':
                print(f"      🤖 LLM chose B: '{val_b}'")
                return val_b
        
        # If no clear A or B, default to A (keep first)
        # print(f"      ⚠️ LLM unclear (got: '{decision}'), defaulting to A")
        return val_a
        
    except Exception as e:
        # print(f"      ❌ LLM judge failed: {e}, defaulting to A")
        return val_a


def hybrid_field_merge(field_name: str, current_val: Any, new_val: Any) -> Any:
    """
    HYBRID: Use fast code rules first, fall back to LLM for complex conflicts.
    Returns the best value between current and new.
    Applies post-processing for student_name to ensure proper formatting.
    """
    # ========================================
    # STEP 1: Handle empty values (instant)
    # ========================================
    if current_val is None or current_val == "":
        result = new_val
    elif new_val is None or new_val == "":
        result = current_val
    else:
        # Convert to strings for comparison
        current_str = str(current_val).strip()
        new_str = str(new_val).strip()
        
        # If identical, no conflict
        if current_str == new_str:
            result = current_val
        
        # ========================================
        # STEP 2: FAST PATH - Code rules (0ms)
        # ========================================
        
        # RULE 1: Exact codes - NEVER change once set
        elif field_name in ["swift_code", "bsb", "iban_number", "account_number"]:
            # print(f"   🔒 {field_name}: CODE RULE - exact codes never change")
            result = current_val
        
        # RULE 2: Numbers - validate integer format
        elif field_name in ["total_tuition_amount", "remit_amount"]:
            try:
                current_int = int(current_val)
                try:
                    int(new_val)
                    # Both are valid integers - keep first
                    # print(f"   💰 {field_name}: CODE RULE - both valid, keeping {current_int}")
                    result = current_val
                except:
                    # Current is valid, new is not
                    # print(f"   💰 {field_name}: CODE RULE - keeping valid {current_int}")
                    result = current_val
            except:
                # Current not valid, try new
                try:
                    new_int = int(new_val)
                    # print(f"   💰 {field_name}: CODE RULE - new is valid {new_int}")
                    result = new_val
                except:
                    # Neither valid, keep first
                    result = current_val
        
        # RULE 3: Obvious garbage in student name
        elif field_name == "student_name":
            unwanted = ["student id", "id:", "dob:", "date of birth", "email", "@", "program"]
            
            new_lower = new_str.lower()
            current_lower = current_str.lower()
            
            new_has_garbage = any(p in new_lower for p in unwanted)
            current_has_garbage = any(p in current_lower for p in unwanted)
            
            # If new has garbage but current doesn't
            if new_has_garbage and not current_has_garbage:
                # print(f"   ⚠️ {field_name}: CODE RULE - rejected new (contains unwanted text)")
                result = current_val
            # If current has garbage but new doesn't
            elif current_has_garbage and not new_has_garbage:
                # print(f"   ✅ {field_name}: CODE RULE - new is cleaner")
                result = new_val
            # If new contains numbers but current doesn't
            elif any(c.isdigit() for c in new_str) and not any(c.isdigit() for c in current_str):
                # print(f"   ⚠️ {field_name}: CODE RULE - rejected new (contains numbers)")
                result = current_val
            else:
                # Fall through to LLM judge
                # print(f"   🤔 {field_name}: CONFLICT detected - asking LLM judge...")
                result = llm_judge(field_name, current_val, new_val)
        
        # RULE 4: Substring detection (university/course names)
        elif field_name in ["university_name", "course_name", "bank_name"]:
            # If current is substring of new (new has extra text)
            if current_str in new_str and len(current_str) < len(new_str):
                # print(f"   ✂️ {field_name}: CODE RULE - keeping shorter '{current_val}'")
                result = current_val
            # If new is substring of current (current has extra text)
            elif new_str in current_str and len(new_str) < len(current_str):
                # print(f"   ✂️ {field_name}: CODE RULE - new is cleaner '{new_val}'")
                result = new_val
            else:
                # Fall through to LLM judge
                # print(f"   🤔 {field_name}: CONFLICT detected - asking LLM judge...")
                result = llm_judge(field_name, current_val, new_val)
        
        # RULE 5: Currency - prefer 3-letter code
        elif field_name == "remit_currency":
            # Check if new is a valid 3-letter code
            if len(new_str) == 3 and new_str.isalpha() and new_str.isupper():
                if not (len(current_str) == 3 and current_str.isalpha() and current_str.isupper()):
                    # print(f"   💱 {field_name}: CODE RULE - '{new_val}' is standard format")
                    result = new_val
                else:
                    result = current_val
            else:
                # Keep current if it's already a valid code
                result = current_val
        
        # RULE 6: Address - prefer longer (more complete)
        elif field_name == "university_address":
            if len(new_str) > len(current_str) * 1.3:  # Significantly longer
                # print(f"   📍 {field_name}: CODE RULE - new is more complete")
                result = new_val
            else:
                result = current_val
        
        # RULE 7: Payment purpose - prefer specific over generic
        elif field_name == "payment_purpose":
            # Generic values that might be defaults/hallucinations
            generic_purposes = ["tuition fee", "tuition", "fee"]
            
            current_lower = current_str.lower()
            new_lower = new_str.lower()
            
            current_is_generic = any(g == current_lower for g in generic_purposes)
            new_is_generic = any(g == new_lower for g in generic_purposes)
            
            # Prefer specific over generic
            if current_is_generic and not new_is_generic:
                # print(f"   💡 {field_name}: CODE RULE - new is more specific '{new_val}'")
                result = new_val
            elif new_is_generic and not current_is_generic:
                # print(f"   💡 {field_name}: CODE RULE - keeping specific '{current_val}'")
                result = current_val
            # Both specific or both generic - prefer longer (more detailed)
            elif len(new_str) > len(current_str):
                # print(f"   💡 {field_name}: CODE RULE - new is more detailed '{new_val}'")
                result = new_val
            else:
                result = current_val
        
        # ========================================
        # STEP 3: SLOW PATH - LLM judge for ambiguous cases (~1.5s)
        # ========================================
        else:
            # print(f"   🤔 {field_name}: CONFLICT detected - asking LLM judge...")
            result = llm_judge(field_name, current_val, new_val)
    
    # ========================================
    # STEP 4: POST-PROCESSING - Apply field-specific formatting
    # ========================================
    if field_name == "student_name" and result:
        cleaned = clean_student_name(result)
        if cleaned != result:
            pass  # print(f"   ✨ POST-PROCESS: '{result}' → '{cleaned}'")
        return cleaned
    
    return result


# ========================================
# API ENDPOINTS
# ========================================

@app.post("/extract-text-image")
async def extract_text_image(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this image"),
):
    """Extract text from a single image file."""
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
    """Extract text from a PDF file (page by page)."""
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
            pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
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
    """Extract structured data from Nepal National ID Card."""
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
        cleaned_response = cleaned_response.replace("``````", "").strip()

        try:
            extracted_json = json.loads(cleaned_response)
            # Handle double-encoded JSON
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
    """Extract structured data from student offer letter with hybrid merge logic."""
    try:
        content = await file.read()
        is_pdf = file.filename.lower().endswith(".pdf")

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
    "student_name": "",
    "course_name": "",
    "university_name": "",
    "university_address": "",
    "total_tuition_amount": null,
    "remit_amount": null,
    "remit_currency": "",
    "iban_number": "",
    "swift_code": "",
    "bsb": "",
    "account_number": "",
    "bank_name": "",
    "payment_purpose": ""
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
- swift_code: SWIFT/BIC code of the beneficiary bank. May be labeled as "SWIFT:", "SWIFT CODE:", "BIC:", "BIC CODE:", or "SWIFT/BIC:". Format is 8-11 characters like 'ABCDAU2SXXX'.
- bsb: BSB code for Australian bank transfers. Look for a 6-digit number often formatted as XXX-XXX (e.g., 062-000). Must return this if found.
- account_number: Bank account number of the beneficiary. Return only if you see account number in the offer letter.
- bank_name: Official name of the beneficiary bank handling the payment. eg. 'Commonwealth Bank of Australia'
- payment_purpose: Purpose of the payment EXACTLY as stated in the document. Common examples include 'Tuition fee deposit', 'COE deposit', 'Living expenses', 'Accommodation fee', 'Application fee', 'Enrollment fee', etc. EXTRACT THE EXACT PHRASE from the document - do NOT assume it's tuition if it says something else.

General rules:
- STRICTLY EXTRACT VALUES you see in the document. Do NOT return null if a value is present.
- Look for bank details (BSB, IBAN, SWIFT, Account Number) in headers, footers, and payment instruction sections.
- BSB codes usually look like 'BSB: 123-456' or just '123-456'.
- SWIFT codes: Extract 8 or 11 character codes formatted like 'ABCDAU2SXXX'. Eg: 'SWIFT Code: WPACAU2SXXX'.
- Copy names, codes, and addresses exactly as written.
- For numeric fields, output plain integers only.
- Do NOT add, remove, or rename any keys in the JSON.

Return ONLY the JSON object.
"""

        # Get images
        if is_pdf:
            pil_images = get_pdf_images(content)
        else:
            pil_images = [Image.open(io.BytesIO(content)).convert("RGB")]

        chunks = list(chunk_list(pil_images, chunk_size=1))
        # print(f"🚀 Processing {len(chunks)} chunks...")

        # COLLECT ALL CHUNKS FIRST
        all_chunks_data = []
        for i, chunk in enumerate(chunks, start=1):
            try:
                print(f"\n📄 Processing chunk {i}/{len(chunks)}...")
                stitched_bytes = stitch_images(chunk)
                
                raw_response = call_gpu_router_ocr(
                    structured_prompt,
                    stitched_bytes,
                    f"offer_letter_chunk_{i}.jpg",
                )
                
                # Clean JSON response
                cleaned_response = (
                    raw_response.replace("```json", "")
                               .replace("```", "")
                               .replace("``````", "")
                               .strip()
                )
                
                chunk_json = json.loads(cleaned_response)
                if isinstance(chunk_json, str):
                    chunk_json = json.loads(chunk_json)
                    
                chunk_data = chunk_json.get("extracted_data", {})
                
                # Store complete chunk data
                all_chunks_data.append({
                    "chunk_num": i,
                    "full_data": chunk_data.copy(),
                    "keys_found": list(chunk_data.keys())
                })
                
                # Debug output
                print(f"✅ Chunk {i}: {list(chunk_data.keys())}")
                for key, value in chunk_data.items():
                    if value:
                        print(f"   {key}: '{value}'")
                        
            except Exception as e:
                pass  # print(f"❌ Chunk {i} failed: {e}")
                continue

        print(f"\n📊 SUMMARY: {len(all_chunks_data)} successful chunks")

        # HYBRID MERGE
        print("\n🔄 HYBRID MERGING...")
        merged_data = {field: None for field in OfferLetterData.__fields__}
        
        for chunk in all_chunks_data:
            chunk_num = chunk["chunk_num"]
            chunk_data = chunk["full_data"]
            
            # print(f"\nMerging chunk {chunk_num}...")
            for key, val in chunk_data.items():
                if val is not None and val != "":
                    current_val = merged_data.get(key)
                    # Use hybrid merge function
                    merged_data[key] = hybrid_field_merge(key, current_val, val)


        # Final result
        print("\n🎉 FINAL MERGED DATA:")
        for key, value in merged_data.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")

        # ========================================
        # POST-PROCESSING: Transliterate non-Latin scripts
        # ========================================
        # print("\n🌐 POST-PROCESSING: Transliterating non-Latin scripts...")
        fields_to_transliterate = ["student_name", "university_name", "bank_name", "university_address"]
        
        for field in fields_to_transliterate:
            if field in merged_data and merged_data[field]:
                original = merged_data[field]
                transliterated = transliterate_to_latin(original, field)
                merged_data[field] = transliterated

        final_structure = {
            "document_type": "offer_letter",
            "extracted_data": merged_data,
        }

        # Validation
        try:
            validated = OfferLetterExtraction(**final_structure)
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated.document_type,
                "extracted_data": validated.extracted_data.dict(),
                "validated": True,
                "raw_json": final_structure,
                "debug_chunks": len(all_chunks_data)
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
    """Health check endpoint to verify API and GPU router connectivity."""
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
