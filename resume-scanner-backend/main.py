from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
import requests
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

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

HF_MODEL_DEFAULT = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Ordered fallbacks (small -> larger, aiming for public Inference API availability)
HF_MODEL_FALLBACKS = [
    os.getenv("HF_MODEL", HF_MODEL_DEFAULT),
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

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
                             model: str = HF_MODEL_DEFAULT, timeout: int = 120) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set in environment")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": True
        }
    }
    logger.info(f"Calling Hugging Face model {model} at {url}")
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    if resp.status_code >= 400:
        # Include short body for diagnosis, but avoid dumping full content
        snippet = (resp.text or "").strip()[:500]
        raise requests.HTTPError(f"HF request failed {resp.status_code} for {model}: {snippet}")
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

@app.get("/")
def root():
    return {"message": "Resume Scanner API is running", "status": "healthy"}


def analyze_core(resume: str, job_desc: str) -> AIAnalyzeOut:
    # Guardrails
    if not resume.strip() or not job_desc.strip():
        return AIAnalyzeOut(missing_skills=[], improvement_suggestions=["Provide both resume and job description."])

    prompt = build_prompt(resume, job_desc)

    # Prefer HF if token provided; otherwise fallback to heuristic
    if HF_API_TOKEN:
        logger.info("HF_API_TOKEN present — attempting AI analysis")
        # Try each candidate model until one succeeds
        last_err: Optional[Exception] = None
        for model in HF_MODEL_FALLBACKS:
            try:
                text = call_hf_text_generation(prompt, max_new_tokens=500, temperature=0.7, model=model)
                return parse_ai_json(text)
            except Exception as e:
                last_err = e
                logger.warning(f"Model failed: {model} — {e}")
                continue
        logger.error("All HF models failed — falling back to heuristic")
        return heuristic_analysis(resume, job_desc)
    else:
        logger.info("HF_API_TOKEN not set — using heuristic analysis")
        return heuristic_analysis(resume, job_desc)


@app.post("/analyze", response_model=AIAnalyzeOut)
def analyze(inp: AnalyzeIn):
    return analyze_core(inp.resume, inp.job_description)


@app.get("/analyze", response_model=AIAnalyzeOut)
def analyze_get(resume: str = "", job_description: str = ""):
    return analyze_core(resume, job_description)
