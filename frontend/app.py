import streamlit as st
import requests
from PIL import Image
import fitz  # PyMuPDF
import io
import json

# ============================
# Page config
# ============================
st.set_page_config(
    page_title="Document Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Extractor")

# ============================
# Sidebar: mode & configuration
# ============================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Choose between unstructured and structured extraction
    mode = st.radio(
        "Extraction mode",
        ["Unstructured text", "Structured JSON"],
        index=0,
    )

    if mode == "Unstructured text":
        # Prompt for unstructured extraction
        unstructured_prompt = st.text_area(
            "Custom Prompt",
            value=(
                "Output only the exact text from this document, "
                "without any explanation or commentary."
            ),
            height=100,
            help="Customize how you want the text to be extracted.",
        )
    else:
        # Structured JSON mode: choose specific document schema
        doc_type = st.selectbox(
            "Structured document type",
            ["national_id", "offer_letter"],
            index=0,
            help="Choose which schema/Pydantic model should be used on the backend.",
        )

st.markdown("---")

# ============================
# Main Layout
# ============================
col1, col2 = st.columns(2)

# ----------------------------
# Left column: Upload & preview
# ----------------------------
with col1:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a document...",
        type=["png", "jpg", "jpeg", "bmp", "pdf"],
        help="Upload an image or PDF document.",
    )

    if uploaded_file:
        filename_lower = uploaded_file.name.lower()

        # PDF preview
        if filename_lower.endswith(".pdf"):
            try:
                pdf_bytes = uploaded_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    st.image(
                        image,
                        caption=f"PDF Preview (Page 1 of {len(doc)})",
                        use_container_width=True,
                    )
                else:
                    st.error("Empty PDF file.")
            except Exception as e:
                st.error(f"Error previewing PDF: {e}")
        else:
            # Image preview
            try:
                image = Image.open(uploaded_file)
                st.image(
                    image, caption="Uploaded Document", use_container_width=True
                )
            except Exception as e:
                st.error(f"Error loading image: {e}")

# ----------------------------
# Right column: Extraction results
# ----------------------------
with col2:
    if mode == "Unstructured text":
        st.subheader("📝 Extracted Text")

        if uploaded_file:
            if st.button("🚀 Extract Text", type="primary"):
                with st.spinner("Extracting text... This may take a few seconds"):
                    try:
                        filename_lower = uploaded_file.name.lower()

                        # Choose backend endpoint based on file type
                        if filename_lower.endswith(".pdf"):
                            api_url = "http://localhost:8000/extract-text-pdf"
                        else:
                            api_url = "http://localhost:8000/extract-text-image"

                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        data = {"prompt": unstructured_prompt}

                        response = requests.post(api_url, files=files, data=data)

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success("✅ Text extracted successfully!")
                                extracted = result["extracted_text"]

                                st.text_area(
                                    "Result:",
                                    value=extracted,
                                    height=400,
                                )

                                st.download_button(
                                    label="📥 Download as TXT",
                                    data=extracted,
                                    file_name=f"{result['filename']}_extracted.txt",
                                    mime="text/plain",
                                )
                            else:
                                st.error(
                                    f"❌ Error: {result.get('error', 'Unknown error')}"
                                )
                        else:
                            st.error(f"❌ HTTP Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a document to extract text.")

    else:
        st.subheader("📊 Structured Output")

        if uploaded_file:
            if st.button("🚀 Extract Structured Data", type="primary"):
                with st.spinner("Extracting and validating structured data..."):
                    try:
                        uploaded_file.seek(0)
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }

                        # Choose backend endpoint based on chosen structured document type
                        if doc_type == "national_id":
                            api_url = "http://localhost:8000/extract-national-id"
                        elif doc_type == "offer_letter":
                            api_url = "http://localhost:8000/extract-offer-letter"
                        else:
                            st.error(f"Unsupported document type: {doc_type}")
                            st.stop()

                        response = requests.post(api_url, files=files)

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success("✅ Extraction successful and validated!")

                                st.metric("Document Type", result["document_type"])

                                st.subheader("Extracted Data")
                                st.json(result["extracted_data"])

                                if result.get("validated"):
                                    st.success("✅ Pydantic Validation: PASSED")

                                json_str = json.dumps(
                                    result["raw_json"],
                                    indent=2,
                                    ensure_ascii=False,
                                )
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name=f"{result['filename']}_structured.json",
                                    mime="application/json",
                                )
                            else:
                                st.error("❌ Extraction failed")
                                st.error(
                                    f"Error: {result.get('error', 'Unknown error')}"
                                )
                                if result.get("validation_errors"):
                                    st.subheader("Validation Errors:")
                                    st.json(result["validation_errors"])
                                if result.get("raw_response"):
                                    with st.expander("Show raw AI response"):
                                        st.text(result["raw_response"])
                        else:
                            st.error(f"❌ HTTP Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a document to extract structured data.")

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown(
    """
**How it works**

- Upload any document image or PDF  
- Choose *Unstructured text* or *Structured JSON*  
- The app calls your FastAPI backend (image-only, PDF-only, or schema-specific routes)  
- For structured extraction, the backend validates the JSON using Pydantic models  
"""
)
