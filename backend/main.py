from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import requests
import base64
from typing import Dict, Any, Optional

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
        # Read and encode image
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Default prompt for structured extraction
        if not custom_prompt:
            structured_prompt = """Analyze this document and extract structured information as JSON.
            
Your response MUST be ONLY valid JSON in this exact format:
{
  "document_type": "invoice|receipt|contract|form|letter|other",
  "extracted_data": {
    // Put all key information here as key-value pairs
    // Examples: "invoice_number": "INV-001", "total": 500, "date": "2024-12-18"
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
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.3,  # Lower temperature for more structured output
            "max_tokens": 2048
        }
        
        # Call llama.cpp server
        response = requests.post(LLAMA_SERVER_URL, json=payload)
        result = response.json()
        
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
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
