Run the backend with Hugging Face AI integration

This project can optionally call the Hugging Face Inference API if you provide an environment variable `HF_API_TOKEN`.

Quick steps (PowerShell on Windows):

1) Install dependencies (in an activated virtualenv):

```powershell
pip install fastapi uvicorn requests pydantic
```

2) Set your Hugging Face token and optional model (replace with your token):

```powershell
$env:HF_API_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
# Optional: change model
$env:HF_MODEL = "tiiuae/falcon-7b-instruct"
```

3) Start the server (from repository root):

```powershell
# If resume-scanner-backend is a python package/module, adjust the import path accordingly.
uvicorn resume_scanner_backend.main:app --reload
```

If the module path above doesn't match your layout, you can run the file directly for quick testing (not recommended for production):

```powershell
python .\resume-scanner-backend\main.py
```

4) Test the `/analyze` endpoint (example):

```powershell
curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"resume":"Experienced Python developer...","job_description":"Looking for Python, FastAPI, machine learning"}'
```

Notes:
- Look at the server logs — when HF is used you'll see a log line saying "Calling Hugging Face model ..." and "Received response from HF model (len=...)".
- If `HF_API_TOKEN` is not set or the HF call fails, the service will fall back to a heuristic analysis and log that decision.
- For production, use a proper process manager and secure secret handling (don't set tokens in shell history).
