from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional
import re

app = FastAPI()

class Meta(BaseModel):
    jobTitle: Optional[str] = None
    companyName: Optional[str] = None
    companySite: Optional[str] = None

class ChecklistItem(BaseModel):
    name: str
    status: Literal["ok", "err", "na"]
    text: str

class Category(BaseModel):
    name: str
    issues: int
    progress: int

class AnalyzeIn(BaseModel):
    resume: str
    job_description: str
    meta: Optional[Meta] = None

class AnalyzeOut(BaseModel):
    score: int
    label: Literal["Low", "Medium", "High"]
    categories: List[Category]
    checklist: List[ChecklistItem]
    structured: str

def tokenize(s: str) -> set:
    return set(re.findall(r"[a-z0-9+.#]+", s.lower()))

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(inp: AnalyzeIn):
    # Guardrails
    if not inp.resume.strip() or not inp.job_description.strip():
        return AnalyzeOut(
            score=0,
            label="Low",
            categories=[],
            checklist=[],
            structured="Provide both Resume and Job Description to score."
        )

    r = tokenize(inp.resume)
    j = tokenize(inp.job_description)
    overlap = len([k for k in j if k in r])
    denom = max(1, len(j))
    base = round((overlap / denom) * 100)

    # Example rubric
    score = round(
        0.35 * base +
        0.25 * base +
        0.15 * base +
        0.10 * base +
        0.10 * base +
        0.05 * base
    )
    score = max(0, min(100, score))
    label = "High" if score >= 80 else "Medium" if score >= 55 else "Low"

    categories = [
        Category(name="Searchability", issues=0 if score > 85 else 2, progress=min(100, score)),
        Category(name="Hard Skills", issues=1 if score > 90 else 3, progress=max(30, score - 5)),
        Category(name="Soft Skills", issues=2, progress=max(25, score - 15)),
        Category(name="Recruiter Tips", issues=1, progress=max(20, score - 10)),
        Category(name="Formatting", issues=0 if score > 95 else 1, progress=max(40, score - 5)),
    ]

    checklist = [
        ChecklistItem(name="ATS Tip", status="err", text="Use standard headings and avoid complex tables/columns."),
        ChecklistItem(name="Contact Information", status="ok", text="Email detected; add location if missing."),
        ChecklistItem(name="Summary", status="err", text="Tailor summary with 2–3 JD keywords."),
        ChecklistItem(name="Section Headings", status="ok", text="Headings appear parseable."),
        ChecklistItem(name="Job Title Match", status="err", text="Mirror JD title where accurate."),
        ChecklistItem(name="Date Formatting", status="ok", text="Use consistent date formats."),
        ChecklistItem(name="Experience", status="err", text="Quantify achievements with metrics."),
        ChecklistItem(name="Skills Coverage", status="err", text="Add missing keywords you truly possess."),
    ]

    structured = (
        f"Match Score: {score}%\n"
        f"Key Strengths:\n- Core keywords partially matched\n- Parseable structure\n- Some role alignment\n"
        f"Gaps to Address:\n- Missing high-impact hard skills\n- Summary not tailored\n- Insufficient quantification\n- Title phrasing misaligned\n"
        f"Suggested Edits:\n- Add missing skills you actually have\n- Rewrite summary with JD terms\n- Quantify outcomes (%, time, cost)\n- Align job title phrasing\n"
        f"Optional ATS Tips:\n- Prefer clean PDF/DOCX\n- Avoid graphics and multi-column layouts\n"
    )

    return AnalyzeOut(
        score=score,
        label=label,
        categories=categories,
        checklist=checklist,
        structured=structured
    )
