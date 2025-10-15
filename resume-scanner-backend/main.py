from fastapi import FastAPI
import logging
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import requests

app = FastAPI()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change ["*"] to specific domains in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Meta(BaseModel):
    jobTitle: Optional[str] = None
    companyName: Optional[str] = None
    companySite: Optional[str] = None

class AnalyzeIn(BaseModel):
    resume: str
    job_description: str
    meta: Optional[Meta] = None

class AIAnalyzeOut(BaseModel):
    missing_skills: List[str]
    improvement_suggestions: List[str]

HF_MODEL_DEFAULT = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

TOKEN_REGEX = re.compile(r"[a-z0-9+.#]+")

def tokenize(s: str) -> set:
    return set(TOKEN_REGEX.findall(s.lower()))

def build_prompt(resume: str, job_desc: str) -> str:
    return (
        "You are an expert resume analyst. Given a resume and a job description, "
        "identify truly missing skills (avoid duplicates and generic words) and provide "
        "specific, actionable improvement suggestions.\n\n"
        "Return ONLY valid JSON with the following exact schema and keys: \n"
        "{\n"
        "  \"missing_skills\": [\"skill\", ...],\n"
        "  \"improvement_suggestions\": [\"suggestion\", ...]\n"
        "}\n\n"
        f"Resume:\n'''\n{resume}\n'''\n\n"
        f"Job Description:\n'''\n{job_desc}\n'''\n\n"
        "Notes:\n- Keep lists concise (max 12 items each).\n- Use plain strings only.\n- No markdown, no comments, no extra text."
    )

def call_hf_text_generation(prompt: str, max_new_tokens: int = 500, temperature: float = 0.7,
                             model: str = HF_MODEL_DEFAULT, timeout: int = 60) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set in environment")
    url = f"https://api-inference.huggingface.co/models/{model}"
    logger.info("Calling Hugging Face model %s at %s", model, url)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # HF can return in two shapes depending on pipeline
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # Some models return {"outputs": [...]}
    if isinstance(data, dict) and "outputs" in data and isinstance(data["outputs"], list):
        return "".join(map(str, data["outputs"]))
    # Fallback to raw string
    return json.dumps(data)

def _strip_code_fences(s: str) -> str:
    # Remove triple backtick fences if present
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[1]
        if "```" in s:
            s = s.split("```", 1)[0]
    return s.strip()

def parse_ai_json(text: str) -> AIAnalyzeOut:
    raw = _strip_code_fences(text)
    # Attempt direct JSON parse
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object heuristically
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        obj = json.loads(m.group(0))
    missing = obj.get("missing_skills") or []
    suggestions = obj.get("improvement_suggestions") or []
    # Coerce to list[str]
    if not isinstance(missing, list):
        missing = [str(missing)]
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]
    missing = [str(x).strip() for x in missing if str(x).strip()]
    suggestions = [str(x).strip() for x in suggestions if str(x).strip()]
    return AIAnalyzeOut(missing_skills=missing, improvement_suggestions=suggestions)

def heuristic_analysis(resume: str, job_desc: str) -> AIAnalyzeOut:
    r = tokenize(resume)
    j = tokenize(job_desc)
    # Very rough heuristic: tokens in JD not in resume, filter out short/common tokens
    STOP = {"and", "or", "the", "a", "an", "with", "in", "for", "to", "of", "on", "at", "by", "as"}
    candidates = [t for t in j if t not in r and len(t) > 2 and t not in STOP]
    # Deduplicate while preserving order
    seen = set()
    missing = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            missing.append(t)
        if len(missing) >= 12:
            break
    suggestions = [
        "Tailor your summary with 2–3 keywords from the job description.",
        "Quantify achievements (%, cost/time saved, revenue impact).",
        "Add relevant hard skills you actually possess and mirror JD phrasing."
    ]
    return AIAnalyzeOut(missing_skills=missing, improvement_suggestions=suggestions)

@app.post("/analyze", response_model=AIAnalyzeOut)
def analyze(inp: AnalyzeIn):
    # Guardrails
    if not inp.resume.strip() or not inp.job_description.strip():
        return AIAnalyzeOut(missing_skills=[], improvement_suggestions=["Provide both resume and job description."])

    prompt = build_prompt(inp.resume, inp.job_description)

    # Prefer HF if token provided; otherwise fallback to heuristic
    if HF_API_TOKEN:
        try:
            logger.info("HF_API_TOKEN present — attempting AI analysis")
            text = call_hf_text_generation(prompt, max_new_tokens=500, temperature=0.7)
            logger.info("Received response from HF model (len=%d)", len(text) if isinstance(text, str) else 0)
            return parse_ai_json(text)
        except Exception:
            logger.exception("HF model call or parsing failed — falling back to heuristic")
            # Fall back to heuristic if API errors or parsing fails
            return heuristic_analysis(inp.resume, inp.job_description)
    else:
        logger.info("HF_API_TOKEN not set — using heuristic analysis")
        return heuristic_analysis(inp.resume, inp.job_description)
