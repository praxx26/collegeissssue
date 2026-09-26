import os
import sys
import streamlit as st
import requests
from PIL import Image

# Ensure local imports work regardless of working directory
sys.path.append(os.path.join(os.path.dirname(__file__), "campus_ai"))

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus AI - Issue Detection",
    page_icon="🏫",
    layout="wide"
)

@st.cache_resource
def load_direct_pipeline():
    """Cached loader for standalone/direct inference mode on free hosting."""
    try:
        from campus_ai.inference import CampusAI
    except ImportError:
        from inference import CampusAI

    model_path = os.path.join(os.path.dirname(__file__), "campus_ai", "models", "best.pt")
    return CampusAI(yolo_model_path=model_path)

def main():
    st.title("🏫 Campus AI Issue Detector")
    st.markdown("Upload a campus image to detect issue classes (garbage, water stagnation, crowd, etc.) and generate an AI summary caption.")

    st.sidebar.header("⚙️ Execution Mode")
    mode = st.sidebar.radio(
        "Select Inference Backend:",
        ["Direct In-Process Model (Cloud / Standalone)", "FastAPI HTTP Server"]
    )

    api_url = "http://localhost:8000/predict"
    if mode == "FastAPI HTTP Server":
        api_url = st.sidebar.text_input("FastAPI Endpoint URL:", value="http://localhost:8000/predict")
        st.sidebar.info("Ensure your FastAPI server is running (`python campus_ai/api.py`).")
    else:
        st.sidebar.success("Running models directly in-process. Perfect for Hugging Face Spaces & Streamlit Cloud!")

    uploaded_file = st.file_uploader("📤 Upload a Campus Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Please upload an image to test the detection pipeline.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Unable to open image: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    result = None

    if mode == "Direct In-Process Model (Cloud / Standalone)":
        with st.spinner("Loading models & running Campus AI inference... (First run downloads weights if needed)"):
            try:
                pipeline = load_direct_pipeline()
                result = pipeline.predict(image)
            except Exception as e:
                st.error(f"Inference Error: {e}")
                return
    else:
        with st.spinner(f"Sending image to FastAPI backend at {api_url}..."):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(api_url, files=files, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                else:
                    st.error(f"Backend API Error ({response.status_code}): {response.text}")
                    return
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to the FastAPI server!")
                st.warning(f"Unable to reach `{api_url}`. Make sure your backend server is running.")
                return
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return

    if result:
        with col2:
            st.subheader("🎯 Detection Output")
            
            # Display caption
            st.success(f"**Caption:** {result.get('caption')}")
            
            # Display predicted classes
            predicted_classes = result.get("predicted_classes", [])
            st.markdown("**Predicted Classes:**")
            if "no issues detected" in predicted_classes:
                st.info("✅ No issues detected")
            else:
                for cls in predicted_classes:
                    st.markdown(f"- 🔴 **{cls}**")

        # Raw Output
        st.divider()
        st.subheader("📊 Raw JSON Payload")
        st.json(result)

if __name__ == "__main__":
    main()