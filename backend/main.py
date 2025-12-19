from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import requests
import base64
from typing import Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image
import io

app = FastAPI(title="Qwen3-VL Text Extractor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED EXTRACTION
# ============================================================================

class DynamicExtraction(BaseModel):
    """Dynamic schema that works with any document type"""
    document_type: str
    extracted_data: Dict[str, Any]
    confidence: Optional[str] = "medium"

# ============================================================================
# EXISTING ENDPOINT (Keep this)
# ============================================================================

@app.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this image")
):
    """Extract unstructured text from uploaded image"""
    
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    response = requests.post(LLAMA_SERVER_URL, json=payload)
    result = response.json()
    
    extracted_text = result['choices'][0]['message']['content']
    
    return {
        "success": True,
        "extracted_text": extracted_text,
        "filename": file.filename
    }

# ============================================================================
# NEW ENDPOINT - STRUCTURED EXTRACTION
# ============================================================================

@app.post("/extract-structured")
async def extract_structured(
    file: UploadFile = File(...),
    custom_prompt: str = Form(None)
):
    """
    Extract structured data from image and return validated JSON
    Uses Pydantic to validate the output structure
    """
    
    try:
        # Universal Image Standardization
        # Goal: Convert everything (PDF page or Image file) to a SINGLE, stitched vertical JPEG
        
        # Read file content
        content = await file.read()
        pil_images = []
        
        if file.filename.lower().endswith('.pdf'):
            # Handle PDF (Multi-page)
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                if len(doc) > 0:
                    print(f"Processing PDF with {len(doc)} pages...")
                    for page_num, page in enumerate(doc):
                        # Get full resolution first
                        mat = fitz.Matrix(2.0, 2.0) # Higher quality for text
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL and append
                        img_data = pix.tobytes("png")
                        pil_images.append(Image.open(io.BytesIO(img_data)).convert('RGB'))
                        print(f"Captured page {page_num+1}")
                else:
                    raise ValueError("Empty PDF file")
            except Exception as e:
                print(f"PDF Error: {e}")
                raise ValueError(f"Failed to process PDF: {str(e)}")
        else:
            # Handle Regular Image
            try:
                pil_images.append(Image.open(io.BytesIO(content)).convert('RGB'))
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")

        if not pil_images:
            raise ValueError("No images processed")
            
        # STITCH IMAGES VERTICALLY
        total_width = max(img.width for img in pil_images)
        total_height = sum(img.height for img in pil_images)
        
        # Create blank canvas
        stitched_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in pil_images:
            # Center the image
            x_offset = (total_width - img.width) // 2
            stitched_image.paste(img, (x_offset, y_offset))
            y_offset += img.height
            
        print(f"Stitched image size: {total_width}x{total_height}")
        
        # Resize if too large (Max height ~3000px to be safe with context)
        MAX_HEIGHT = 3000
        if total_height > MAX_HEIGHT:
            ratio = MAX_HEIGHT / total_height
            new_width = int(total_width * ratio)
            stitched_image = stitched_image.resize((new_width, MAX_HEIGHT), Image.Resampling.LANCZOS)
            print(f"Resized to {new_width}x{MAX_HEIGHT}")

        # Convert to standard JPEG
        output_buffer = io.BytesIO()
        stitched_image.save(output_buffer, format='JPEG', quality=85)
        final_image_bytes = output_buffer.getvalue()
        
        # Encode
        image_base64 = base64.b64encode(final_image_bytes).decode('utf-8')
        mime_type = "image/jpeg"
        print(f"Final payload size: {len(final_image_bytes)} bytes (JPEG)")
        
        # Default prompt for structured extraction
        if not custom_prompt:
            structured_prompt = """Analyze ALL provided images (pages of the document) and extract structured information as JSON.
            
Your response MUST be ONLY valid JSON in this exact format:
{
  "document_type": "invoice|receipt|contract|form|letter|other",
  "extracted_data": {
    // Put all key information here as key-value pairs.
    // Ensure you extract information from ALL pages.
    // Examples: "invoice_number": "INV-001", "total": 500, "date": "2024-12-18", "items": [...]
  },
  "confidence": "high|medium|low"
}

Return ONLY the JSON, no explanations, no markdown, no code blocks."""
        else:
            structured_prompt = custom_prompt
        
        # Prepare request for llama.cpp
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": structured_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,  # Lower temperature for more structured output
            "max_tokens": 2048
        }
        
        # Call llama.cpp server
        print(f"Sending request to LLM server... (Image: {len(image_base64)} chars)")
        response = requests.post(LLAMA_SERVER_URL, json=payload)
        
        if response.status_code != 200:
            print(f"LLM Server Error: {response.status_code} - {response.text}")
            raise Exception(f"LLM Server Error: {response.text}")
            
        result = response.json()
        
        if 'choices' not in result:
            print(f"Unexpected LLM response: {result}")
            error_msg = result.get('error', {}).get('message', 'Unknown LLM error') if isinstance(result.get('error'), dict) else str(result)
            raise Exception(f"LLM Error: {error_msg}")
        
        # Get the raw response
        raw_response = result['choices'][0]['message']['content']
        
        # Clean up the response (remove markdown code blocks if present)
        import json
        import re
        
        # Remove markdown code blocks
        cleaned_response = re.sub(r'```json|```', '', raw_response)
        cleaned_response = re.sub(r'```\s*', '', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        # Parse JSON
        try:
            extracted_json = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": "Invalid JSON from model",
                "raw_response": raw_response,
                "validation_error": str(e)
            }
        
        # Validate with Pydantic
        try:
            validated_data = DynamicExtraction(**extracted_json)
            
            return {
                "success": True,
                "filename": file.filename,
                "document_type": validated_data.document_type,
                "extracted_data": validated_data.extracted_data,
                "confidence": validated_data.confidence,
                "validated": True,
                "raw_json": extracted_json
            }
            
        except ValidationError as e:
            return {
                "success": False,
                "error": "Pydantic validation failed",
                "validation_errors": e.errors(),
                "raw_json": extracted_json
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Check if the API and llama.cpp server are running"""
    try:
        response = requests.get("http://localhost:8080/health")
        return {"status": "healthy", "llama_server": "running"}
    except:
        return {"status": "degraded", "llama_server": "offline"}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    print("PDF Support: Enabled (PyMuPDF detected)")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
