from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="Qwen3-VL Text Extractor API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# llama.cpp server URL (your Docker container)
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

@app.post("/extract-text")
async def extract_text(
    file: UploadFile = File(...),
    prompt: str = Form("Output only the exact text from this image, without any explanation or commentary. Just the text content.")
):
    """Extract text from uploaded image using Qwen3-VL"""
    
    # Read and encode image
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare request for llama.cpp server
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
    
    # Call llama.cpp server
    response = requests.post(LLAMA_SERVER_URL, json=payload)
    result = response.json()
    
    extracted_text = result['choices'][0]['message']['content']
    
    return {
        "success": True,
        "extracted_text": extracted_text,
        "filename": file.filename
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
