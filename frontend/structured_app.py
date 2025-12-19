import streamlit as st
import requests
import json
from PIL import Image
import fitz # PyMuPDF
import io

# Page config
st.set_page_config(
    page_title="Structured Data Extractor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Structured Data Extraction with Pydantic")
st.markdown("Extract structured JSON data from documents with automatic validation")

# API endpoint
API_URL = "http://localhost:8000/extract-structured"

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    use_custom_prompt = st.checkbox("Use Custom Prompt")
    
    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Extraction Prompt",
            height=200,
            placeholder="Describe what structure you want..."
        )
    else:
        custom_prompt = None
        st.info("Using default structured extraction prompt")
    
    st.markdown("---")
    st.markdown("### 📝 Example Documents:")
    st.code("✓ Invoices")
    st.code("✓ Receipts")
    st.code("✓ Forms")
    st.code("✓ Contracts")
    st.code("✓ ID Cards")

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a document...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'pdf'],
        help="Upload an image or PDF document"
    )
    
    if uploaded_file:
        if uploaded_file.name.lower().endswith('.pdf'):
            # Handle PDF preview
            try:
                # Create a fresh copy of bytes for preview so we don't consume the main stream
                pdf_bytes = uploaded_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image, caption=f"PDF Preview (Page 1 of {len(doc)})", use_container_width=True)
                else:
                    st.error("Empty PDF")
            except Exception as e:
                st.error(f"Error previewing PDF: {e}")
        else:
            # Handle Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_container_width=True)

with col2:
    st.subheader("📊 Structured Output")
    
    if uploaded_file:
        if st.button("🚀 Extract Structured Data", type="primary"):
            with st.spinner("Extracting and validating structured data..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Prepare request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {}
                    if custom_prompt:
                        data["custom_prompt"] = custom_prompt
                    
                    # Call API
                    response = requests.post(API_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            st.success("✅ Extraction successful and validated!")
                            
                            # Display document type
                            st.metric("Document Type", result['document_type'])
                            st.metric("Confidence", result.get('confidence', 'N/A'))
                            
                            # Display extracted data
                            st.subheader("Extracted Data")
                            st.json(result['extracted_data'])
                            
                            # Show validation status
                            if result.get('validated'):
                                st.success("✅ Pydantic Validation: PASSED")
                            
                            # Download JSON
                            json_str = json.dumps(result['raw_json'], indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_str,
                                file_name=f"{result['filename']}_structured.json",
                                mime="application/json"
                            )
                        else:
                            st.error("❌ Extraction failed")
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                            
                            if result.get('validation_errors'):
                                st.subheader("Validation Errors:")
                                st.json(result['validation_errors'])
                            
                            if result.get('raw_response'):
                                with st.expander("Show raw AI response"):
                                    st.text(result['raw_response'])
                    else:
                        st.error(f"❌ HTTP Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload a document to extract structured data")

# Footer
st.markdown("---")
st.markdown("""
### How it works:
1. **Upload** any document image
2. **AI analyzes** and extracts key information
3. **Returns JSON** with structured data
4. **Pydantic validates** the JSON structure
5. **Download** validated JSON for use in your apps
""")
