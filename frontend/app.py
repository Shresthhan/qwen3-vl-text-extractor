import streamlit as st
import requests
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Qwen3-VL Text Extractor",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("Qwen3-VL Text Extractor")
st.markdown("Upload an image to extract text using local AI")

# API endpoint
API_URL = "http://localhost:8000/extract-text"

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    prompt = st.text_area(
        "Custom Prompt",
        value="Output only the exact text from this image, without any explanation or commentary. Just the text content.",
        height=100,
        help="Customize what you want to extract"
    )

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a document, invoice, receipt, or any image with text"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Extracted Text")
    
    if uploaded_file:
        if st.button("🚀 Extract Text", type="primary"):
            with st.spinner("Extracting text... This may take a few seconds"):
                try:
                    # Prepare file for upload
                    files = {"file": uploaded_file.getvalue()}
                    data = {"prompt": prompt}
                    
                    # Call API
                    response = requests.post(API_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Text extracted successfully!")
                        
                        # Display extracted text
                        extracted = result['extracted_text']
                        st.text_area(
                            "Result:",
                            value=extracted,
                            height=400
                        )
                        
                        # Download button
                        st.download_button(
                            label="📥 Download as TXT",
                            data=extracted,
                            file_name=f"{result['filename']}_extracted.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure the backend API is running on port 8000")
    else:
        st.info("Upload an image to get started")

# Footer
st.markdown("---")
st.markdown(
    "Powered by **Qwen3-VL-2B-Instruct** "
)
