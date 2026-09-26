# 🌐 Free Hosting Guide for Campus AI

This project contains heavy AI models (**PyTorch**, **YOLO11**, **CLIP**, and **BLIP**), which require ~2 GB RAM. Standard free tiers like Render, Vercel, or Netlify free servers have 512 MB RAM limits and will crash with Out Of Memory (OOM) errors.

Here are the **best 100% FREE platforms** to host this project, along with step-by-step instructions.

---

## 🏆 Option 1: Hugging Face Spaces (Recommended - 100% Free)

**Hugging Face Spaces** provides **16 GB RAM + 2 vCPU + 50 GB storage** for **FREE**. It is the absolute best platform for hosting Python AI applications.

### Steps to Deploy on Hugging Face Spaces:

1. **Create a Free Account**:
   - Go to [huggingface.co](https://huggingface.co/) and sign up.

2. **Create a New Space**:
   - Click your profile icon at top right -> **New Space**.
   - **Space Name**: `campus-ai-detector` (or any name you choose).
   - **License**: `MIT` or `apache-2.0`.
   - **Select the Space SDK**: Choose **Streamlit**.
   - **Space Hardware**: Keep **CPU Basic (Free - 16GB RAM)**.
   - **Visibility**: Public.
   - Click **Create Space**.

3. **Upload / Push your Code**:
   - You can push via Git or upload files directly through the Hugging Face web UI:
     - `app.py`
     - `requirements.txt`
     - `campus_ai/` directory (including `campus_ai/models/best.pt`)
     - `.gitignore`

4. **Build & Live**:
   - Hugging Face automatically builds your container and launches Streamlit.
   - You will get a permanent public share link like: `https://huggingface.co/spaces/<your-username>/campus-ai-detector`.

---

## 🥈 Option 2: Streamlit Community Cloud (100% Free)

**Streamlit Community Cloud** allows you to host Streamlit apps directly from your GitHub repository for free.

### Steps to Deploy on Streamlit Cloud:

1. **Push your project to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/campus_issue.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Sign in with your GitHub account.
   - Click **New app**.
   - Select your repository (`campus_issue`), branch (`main`), and main file path (`app.py`).
   - Click **Deploy!**.

---

## 🥉 Option 3: Cloudflare Tunnel / LocalTunnel (Instant Public URL from your PC)

If you prefer to run the app on your computer (using your GPU) while sharing a live HTTPS URL with team members or clients anywhere in the world:

### Using LocalTunnel (No installation required):
1. Run `app.py` locally:
   ```bash
   streamlit run app.py
   ```
2. In a second terminal, expose port 8501 to the web:
   ```bash
   npx localtunnel --port 8501
   ```
3. Share the generated public URL!

---

## 🛠️ Summary of Files Created / Prepared for Cloud Hosting

| File | Purpose |
| :--- | :--- |
| [`app.py`](file:///c:/campus_issue/app.py) | Streamlit UI with dual support (Direct In-Process AI & FastAPI backend) |
| [`requirements.txt`](file:///c:/campus_issue/requirements.txt) | Complete dependency list for cloud builders |
| [`Dockerfile`](file:///c:/campus_issue/Dockerfile) | Container definition for Docker-based free hosts |
| [`.gitignore`](file:///c:/campus_issue/.gitignore) | Keeps repository clean of large training datasets |
| [`campus_ai/api.py`](file:///c:/campus_issue/campus_ai/api.py) | FastAPI backend with dynamic PORT/HOST configuration |
