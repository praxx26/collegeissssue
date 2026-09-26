import streamlit as st
import requests
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus AI API Client",
    page_icon="🏫",
    layout="wide"
)

# API Endpoint URL - Using your machine's Network IP so it can be shared with your team!
API_URL = "http://10.218.107.167:8000/predict"

def main():
    st.title("🏫 Campus AI Client (FastAPI Backend)")
    st.markdown("Upload a campus image. This frontend will send the image securely over HTTP to your FastAPI backend for detection.")

    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"Connecting to API Endpoint: \n`{API_URL}`")
    st.sidebar.markdown("*(Ensure your FastAPI server is running in another terminal)*")

    uploaded_file = st.file_uploader("📤 Upload a Campus Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Please upload an image to test the API connection.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Unable to open image locally: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Sending image to FastAPI backend... (This might take a few seconds for the BLIP model)"):
        # Reset file pointer just in case Streamlit has moved it
        uploaded_file.seek(0)
        
        # Package the file as a multipart/form-data request exactly like the cURL command
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        try:
            # Hit the FastAPI /predict endpoint!
            response = requests.post(API_URL, files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
            else:
                st.error(f"Backend API Error ({response.status_code}): {response.text}")
                return
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Failed to connect to the FastAPI server!")
            st.warning("Please make sure you have started the server in your terminal by running:\n\n`python campus_ai/api.py`")
            return
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return

    with col2:
        st.subheader("🎯 Prediction Received from API")
        
        # Display the caption
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
    st.subheader("📊 Raw JSON Payload (Received from API)")
    st.markdown("This is the exact JSON data that your backend successfully served over the HTTP connection.")
    st.json(result)

if __name__ == "__main__":
    main()