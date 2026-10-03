"""
ScholarshipFit AI — Streamlit MVP

Purpose
-------
Analyze a user's academic/research profile, discover relevant Master's/PhD
opportunities, compare a selected position against the profile, and generate
application documents.

Supported AI providers
----------------------
- OpenAI (Responses API)
- DeepSeek (OpenAI-compatible API)
- Gemini (google-genai)
- Anthropic (Messages API)

Notes
-----
1. The app keeps uploaded/profile data in Streamlit session state only.
2. API keys can be entered in the sidebar for a prototype or loaded from
   Streamlit secrets / environment variables.
3. Position discovery uses the selected AI provider's native web-search tool.
4. Search results are "leads" and should be verified on the official
   university/employer page before application.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

# Optional provider SDKs are included in requirements.txt.
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic


# =============================================================================
# App configuration
# =============================================================================

st.set_page_config(
    page_title="ScholarshipFit AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "ScholarshipFit AI"
APP_SUBTITLE = "AI-assisted Master's & PhD scholarship matching and application preparation"

DEFAULT_MODELS = {
    "OpenAI": "gpt-6-luna",
    "DeepSeek": "deepseek-flash",
    "Gemini": "gemini-3.8-flash",
    "Anthropic": "claude-sonnet-4-5",
}

# Gemini fallback order for temporary capacity / availability problems.
GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    # "gemini-2.5-flash",
]

PROVIDER_KEYS = {
    "OpenAI": ["OPENAI_API_KEY"],
    "DeepSeek": ["DEEPSEEK_API_KEY"],
    "Gemini": ["GEMINI_API_KEY"],
    "Anthropic": ["ANTHROPIC_API_KEY"],
}

MAX_CV_CHARS = 30000
MAX_POSITION_CHARS = 30000
MAX_GENERATED_DOC_CHARS = 50000


# =============================================================================
# Session state
# =============================================================================

def init_state() -> None:
    defaults = {
        "profile": {},
        "profile_raw": "",
        "positions": [],
        "selected_position": {},
        "position_analysis": {},
        "generated_document": "",
        "generated_document_type": "",
        "search_results_raw": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =============================================================================
# Utility helpers
# =============================================================================

def get_secret(name: str) -> str:
    """Read a secret from Streamlit secrets first, then environment."""
    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        pass

    return os.getenv(name, "")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_loads(value: str) -> Any:
    """Best-effort extraction of JSON from an LLM response."""
    if not value:
        raise ValueError("Empty AI response.")

    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first plausible JSON object or array.
    candidates = [
        (text.find("{"), text.rfind("}")),
        (text.find("["), text.rfind("]")),
    ]
    for start, end in candidates:
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not parse JSON from AI response.")


def stringify_profile(profile: Dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, ensure_ascii=False)


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n\n".join(pages))


def extract_docx_text(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    chunks = [p.text for p in doc.paragraphs if p.text.strip()]

    # Also extract table text because many CVs use tables.
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                chunks.append(row_text)

    return clean_text("\n".join(chunks))


def extract_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".pdf":
        return extract_pdf_text(raw)
    if suffix == ".docx":
        return extract_docx_text(raw)
    if suffix == ".txt":
        return clean_text(raw.decode("utf-8", errors="ignore"))

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def extract_webpage_text(url: str) -> str:
    """Fetch readable text from a position page."""
    url = normalize_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ScholarshipFitAI/1.0; "
            "+https://localhost)"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Prefer main/article content where available.
    container = soup.find("main") or soup.find("article") or soup.body
    text = container.get_text("\n", strip=True) if container else soup.get_text("\n")

    text = clean_text(text)
    return text[:MAX_POSITION_CHARS]


def build_profile_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the structured form into a profile schema."""
    return {
        "name": form_data.get("name", ""),
        "email": form_data.get("email", ""),
        "current_location": form_data.get("current_location", ""),
        "target_degree": form_data.get("target_degree", ""),
        "target_countries": form_data.get("target_countries", ""),
        "education": [
            {
                "degree": form_data.get("degree_1", ""),
                "field": form_data.get("field_1", ""),
                "institution": form_data.get("institution_1", ""),
                "country": form_data.get("country_1", ""),
                "year": form_data.get("year_1", ""),
                "grade": form_data.get("grade_1", ""),
            },
            {
                "degree": form_data.get("degree_2", ""),
                "field": form_data.get("field_2", ""),
                "institution": form_data.get("institution_2", ""),
                "country": form_data.get("country_2", ""),
                "year": form_data.get("year_2", ""),
                "grade": form_data.get("grade_2", ""),
            },
        ],
        "research_domains": form_data.get("research_domains", ""),
        "research_methods": form_data.get("research_methods", ""),
        "publications": form_data.get("publications", ""),
        "experience": form_data.get("experience", ""),
        "technical_skills": form_data.get("technical_skills", ""),
        "english_test": form_data.get("english_test", ""),
        "funding_need": form_data.get("funding_need", ""),
        "preferred_supervisor": form_data.get("preferred_supervisor", ""),
        "other_information": form_data.get("other_information", ""),
    }


# =============================================================================
# AI provider layer
# =============================================================================

def get_api_key(provider: str, sidebar_key: str) -> str:
    if sidebar_key.strip():
        return sidebar_key.strip()

    for env_key in PROVIDER_KEYS.get(provider, []):
        value = get_secret(env_key)
        if value:
            return value

    return ""


def call_ai(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_web_search: bool = False,
) -> str:
    """Call a selected model and return plain text, optionally using native web search."""
    if not api_key:
        raise ValueError(
            f"No API key configured for {provider}. "
            "Enter it in the sidebar or add it to Streamlit secrets/environment."
        )

    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)

        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
        }

        if use_web_search:
            # OpenAI's Responses API supports built-in web search tools.
            kwargs["tools"] = [{"type": "web_search"}]

        response = client.responses.create(**kwargs)
        return response.output_text.strip()

    if provider == "DeepSeek":
        if use_web_search:
            raise ValueError(
                "DeepSeek does not provide built-in web search. "
                "Select OpenAI, Gemini, or Anthropic for live opportunity search."
            )
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "Gemini":
        client = genai.Client(api_key=api_key)

        requested = model.strip() or DEFAULT_MODELS["Gemini"]
        candidates = []
        for candidate in [requested, *GEMINI_FALLBACK_MODELS]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        last_error = None
        for candidate in candidates:
            for attempt in range(3):
                try:
                    config_options: Dict[str, Any] = {
                        "temperature": 0.2,
                        "max_output_tokens": 6000,
                    }
                    if use_web_search:
                        config_options["tools"] = [
                            types.Tool(google_search=types.GoogleSearch())
                        ]
                    response = client.models.generate_content(
                        model=candidate,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(
                                    text=f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
                                         f"USER REQUEST:\n{user_prompt}"
                                )],
                            )
                        ],
                        config=types.GenerateContentConfig(**config_options),
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text
                    raise RuntimeError(f"Gemini returned an empty response using {candidate}.")
                except Exception as exc:
                    last_error = exc
                    message = str(exc).lower()
                    is_capacity_error = (
                        "503" in message
                        or "unavailable" in message
                        or "high demand" in message
                        or "resource exhausted" in message
                        or "429" in message
                    )
                    if is_capacity_error and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    break

        raise RuntimeError(
            "Gemini could not process the request after trying available fallback "
            f"models. Last error: {last_error}"
        )

    if provider == "Anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        tools = (
            [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]
            if use_web_search
            else None
        )
        response = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=6000,
            temperature=0.2,
            messages=[{"role": "user", "content": user_prompt}],
            **({"tools": tools} if tools else {}),
        )
        parts = []
        for item in response.content:
            if getattr(item, "type", "") == "text":
                parts.append(item.text)
        return "\n".join(parts).strip()

    raise ValueError(f"Unsupported provider: {provider}")


# =============================================================================
# Prompt templates
# =============================================================================

PROFILE_EXTRACT_SYSTEM = """
You are an expert academic admissions and scholarship-profile analyst.
Transform the supplied CV/profile text into a structured, evidence-based
candidate profile.

Rules:
- Do not invent degrees, publications, skills, employers, dates, grades, or
  research topics.
- Preserve uncertainty where the source is incomplete.
- Separate evidence from assumptions.
- Return ONLY valid JSON.
"""

PROFILE_EXTRACT_PROMPT = """
Extract this person's information using the following JSON schema:

{
  "name": "",
  "current_role": "",
  "education": [
    {
      "degree": "",
      "field": "",
      "institution": "",
      "country": "",
      "year": "",
      "grade": ""
    }
  ],
  "research_domains": [],
  "research_methods": [],
  "technical_skills": [],
  "publications": [
    {
      "title": "",
      "venue": "",
      "year": "",
      "doi_or_url": ""
    }
  ],
  "experience": [
    {
      "role": "",
      "organization": "",
      "period": "",
      "relevance": ""
    }
  ],
  "tests_and_certifications": [],
  "funding_preferences": [],
  "target_degree": "",
  "target_countries": [],
  "evidence_notes": []
}

SOURCE MATERIAL:
"""


MATCH_SYSTEM = """
You are a graduate-scholarship matching analyst.

Given a candidate profile and a set of Master's/PhD opportunities, assess
compatibility using explicit evidence.

Do NOT claim that any opportunity is guaranteed, "perfect", or likely to be
successful. Use a transparent fit estimate as a heuristic only.

Return ONLY valid JSON.
"""

MATCH_PROMPT = """
Candidate profile:
{profile}

Opportunities:
{positions}

For each opportunity produce:
{
  "matches": [
    {
      "title": "",
      "university": "",
      "country": "",
      "degree": "",
      "funding": "",
      "deadline": "",
      "url": "",
      "fit_score": 0,
      "matching_evidence": [],
      "gaps_or_risks": [],
      "why_relevant": ""
    }
  ]
}

Scoring guidance:
- 90-100: very strong documented alignment
- 75-89: strong alignment with some gaps
- 60-74: plausible but important gaps
- below 60: weaker alignment

The score is a heuristic, not a probability of admission or scholarship success.
"""


POSITION_ANALYSIS_SYSTEM = """
You are an academic admissions strategy analyst.
Compare a candidate against a specific graduate position.

Be strict and evidence-based:
- Identify hard eligibility requirements.
- Identify desirable/research-fit requirements.
- Identify missing evidence.
- Distinguish "unknown" from "not met".
- Suggest concrete improvements that the applicant can make without inventing
  experience.
- Identify every application document mentioned or implied by the position
  advertisement.

Return ONLY valid JSON.
"""

POSITION_ANALYSIS_PROMPT = """
CANDIDATE:
{profile}

POSITION:
{position}

Return:
{
  "overall_fit_summary": "",
  "eligibility": {
    "meets": [],
    "not_met": [],
    "unknown": []
  },
  "research_fit": {
    "strong_matches": [],
    "partial_matches": [],
    "gaps": []
  },
  "skills_fit": {
    "strong_matches": [],
    "missing_or_weak": []
  },
  "application_strategy": [],
  "required_documents": [
    {
      "document": "",
      "required": true,
      "evidence_from_ad": ""
    }
  ],
  "questions_to_clarify": [],
  "recommended_next_steps": []
}
"""


DOCUMENT_SYSTEM = """
You are an academic application-writing specialist.

Write a truthful, evidence-based application document using ONLY facts present
in the candidate profile and position information.

Important:
- Never fabricate achievements, papers, projects, grants, grades, awards,
  supervisors, experiments, dates, or personal experiences.
- Do not overclaim fit.
- Tailor the document to the exact advertised research topic, methods,
  requirements, and requested structure.
- Use clear professional academic English.
- Avoid generic AI-sounding filler.
"""

DOCUMENT_PROMPT = """
Candidate profile:
{profile}

Position advertisement:
{position}

Position analysis:
{analysis}

Document type:
{document_type}

Additional user instructions:
{extra_instructions}

Write the requested document. Do not include commentary before or after it.
"""


RESEARCH_PROPOSAL_SYSTEM = """
You are a research-proposal specialist for funded Master's/PhD applications.

The proposal must be scientifically credible but must not fabricate results.
Build from the candidate's documented background and the advertised research
problem. Mark assumptions as proposed directions.

Use this structure unless the advertisement specifies another:
Title
1. Background and problem
2. Research gap
3. Aim and objectives
4. Research questions / hypotheses
5. Proposed methodology
6. Data / resources
7. Evaluation
8. Expected contribution
9. Risks and alternatives
10. Indicative timeline
11. References or suggested literature areas
"""


# =============================================================================
# Native AI web discovery
# =============================================================================

# =============================================================================
# Position helpers
# =============================================================================

def position_from_fields(
    title: str,
    university: str,
    country: str,
    degree: str,
    funding: str,
    deadline: str,
    url: str,
    description: str,
) -> Dict[str, Any]:
    return {
        "title": title.strip(),
        "university": university.strip(),
        "country": country.strip(),
        "degree": degree.strip(),
        "funding": funding.strip(),
        "deadline": deadline.strip(),
        "url": normalize_url(url),
        "description": description.strip(),
    }


def parse_position_search_response(text: str) -> List[Dict[str, Any]]:
    data = safe_json_loads(text)

    if isinstance(data, dict):
        candidates = data.get("positions") or data.get("matches") or []
    elif isinstance(data, list):
        candidates = data
    else:
        candidates = []

    positions = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        positions.append(
            {
                "title": str(item.get("title", "")),
                "university": str(item.get("university", "")),
                "country": str(item.get("country", "")),
                "degree": str(item.get("degree", "")),
                "funding": str(item.get("funding", "")),
                "deadline": str(item.get("deadline", "")),
                "url": normalize_url(str(item.get("url", ""))),
                "description": str(item.get("description", "")),
            }
        )
    return positions


# =============================================================================
# Word document export
# =============================================================================

def make_docx(title: str, content: str) -> bytes:
    doc = Document()
    doc.add_heading(title, level=1)

    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        # Simple markdown-ish heading support.
        if re.match(r"^#{1,3}\s+", block):
            cleaned = re.sub(r"^#{1,3}\s+", "", block)
            doc.add_heading(cleaned, level=2)
        else:
            for paragraph in block.split("\n"):
                paragraph = paragraph.strip()
                if paragraph:
                    doc.add_paragraph(paragraph)

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("⚙️ AI Settings")

    provider = st.selectbox(
        "AI provider",
        list(DEFAULT_MODELS.keys()),
        index=0,
    )

    api_key = st.text_input(
        f"{provider} API key",
        value="",
        type="password",
        help=(
            "Prototype option: enter the key here. "
            "For deployment, prefer Streamlit secrets/environment variables."
        ),
    )

    model = st.text_input(
        "Model",
        value=DEFAULT_MODELS[provider],
        help="Change this when your provider account uses another available model.",
    )

    st.divider()
    st.caption(
        "For production, store secrets outside source code. "
        "Streamlit supports .streamlit/secrets.toml."
    )

    st.divider()
    st.caption("MVP data model")
    st.caption("Profile → Opportunities → Position Analysis → Documents")


# =============================================================================
# Header
# =============================================================================

st.title(f"🎓 {APP_TITLE}")
st.write(APP_SUBTITLE)

with st.expander("What this MVP does"):
    st.markdown(
        """
        **1. Profile analysis** — Upload a CV or fill the structured profile form.

        **2. Opportunity discovery** — Search for current opportunities (OpenAI
        live web search) or add opportunities manually.

        **3. Position-specific analysis** — Compare a selected advertisement
        against your education, experience, research, skills, and constraints.

        **4. Application documents** — Draft emails, motivation letters,
        statements of interest, SOPs, research proposals, and custom documents.

        **Important:** AI outputs are assistance, not admissions predictions.
        Always verify eligibility, deadline, funding, and application
        instructions on the official page.
        """
    )


# =============================================================================
# Main navigation
# =============================================================================

tabs = st.tabs(
    [
        "1 · Build Profile",
        "2 · Find Positions",
        "3 · Analyze Position",
        "4 · Prepare Documents",
    ]
)


# =============================================================================
# TAB 1 — Profile
# =============================================================================

with tabs[0]:
    st.header("Build your academic profile")

    profile_method = st.radio(
        "Choose input method",
        ["Upload CV", "Structured form", "Both"],
        horizontal=True,
    )

    uploaded_text = ""

    if profile_method in {"Upload CV", "Both"}:
        uploaded_file = st.file_uploader(
            "Upload CV",
            type=["pdf", "docx", "txt"],
            help="PDF and DOCX text extraction is supported. Scanned PDFs may need OCR later.",
        )

        if uploaded_file:
            try:
                uploaded_text = extract_uploaded_file(uploaded_file)
                st.success(
                    f"Extracted {len(uploaded_text):,} characters from {uploaded_file.name}."
                )
                with st.expander("Preview extracted CV text"):
                    st.text_area(
                        "CV text",
                        uploaded_text[:15000],
                        height=300,
                        label_visibility="collapsed",
                    )
            except Exception as exc:
                st.error(f"Could not read the CV: {exc}")

    form_data = {}
    if profile_method in {"Structured form", "Both"}:
        with st.form("profile_form"):
            c1, c2 = st.columns(2)

            with c1:
                form_data["name"] = st.text_input("Full name")
                form_data["email"] = st.text_input("Email")
                form_data["current_location"] = st.text_input("Current location")
                form_data["target_degree"] = st.selectbox(
                    "Target degree",
                    ["PhD", "Master's", "MPhil/MS", "Either"],
                )
                form_data["target_countries"] = st.text_input(
                    "Target countries",
                    placeholder="UK, Germany, Switzerland, Canada",
                )

            with c2:
                form_data["research_domains"] = st.text_input(
                    "Research domains",
                    placeholder="3D vision, medical imaging, robotics",
                )
                form_data["research_methods"] = st.text_input(
                    "Research methods",
                    placeholder="deep learning, point clouds, transformers",
                )
                form_data["technical_skills"] = st.text_input(
                    "Technical skills",
                    placeholder="Python, PyTorch, OpenCV, C#, Flutter",
                )
                form_data["english_test"] = st.text_input(
                    "English test / score",
                    placeholder="IELTS 7.0, TOEFL 100, etc.",
                )
                form_data["funding_need"] = st.selectbox(
                    "Funding need",
                    ["Fully funded required", "Partial funding acceptable", "Self-funded"],
                )

            st.subheader("Education")
            e1, e2 = st.columns(2)

            with e1:
                form_data["degree_1"] = st.text_input("Highest degree")
                form_data["field_1"] = st.text_input("Highest-degree field")
                form_data["institution_1"] = st.text_input("Highest-degree institution")
                form_data["country_1"] = st.text_input("Institution country")
                form_data["year_1"] = st.text_input("Graduation year")
                form_data["grade_1"] = st.text_input("CGPA / grade")

            with e2:
                form_data["degree_2"] = st.text_input("Previous degree")
                form_data["field_2"] = st.text_input("Previous-degree field")
                form_data["institution_2"] = st.text_input("Previous institution")
                form_data["country_2"] = st.text_input("Previous institution country")
                form_data["year_2"] = st.text_input("Previous graduation year")
                form_data["grade_2"] = st.text_input("Previous CGPA / grade")

            form_data["publications"] = st.text_area(
                "Publications",
                placeholder="Title — venue — year — DOI/URL",
                height=120,
            )
            form_data["experience"] = st.text_area(
                "Research / professional experience",
                placeholder="Role — organization — period — responsibilities",
                height=140,
            )
            form_data["preferred_supervisor"] = st.text_input(
                "Preferred supervisor(s), if any"
            )
            form_data["other_information"] = st.text_area(
                "Other relevant information",
                placeholder="Scholarships, awards, thesis details, projects, etc.",
                height=120,
            )

            submitted_form = st.form_submit_button("Save structured profile")

        if submitted_form:
            st.session_state["profile"] = build_profile_from_form(form_data)
            st.session_state["profile_raw"] = json.dumps(form_data, indent=2)
            st.success("Structured profile saved.")

    st.divider()

    # AI extraction can be done from CV or from a form snapshot.
    extraction_source = uploaded_text if uploaded_text else st.session_state.get("profile_raw", "")

    if extraction_source:
        if st.button("🤖 Analyze / normalize profile with AI", type="primary"):
            try:
                source = extraction_source[:MAX_CV_CHARS]
                result = call_ai(
                    provider=provider,
                    api_key=get_api_key(provider, api_key),
                    model=model,
                    system_prompt=PROFILE_EXTRACT_SYSTEM,
                    user_prompt=PROFILE_EXTRACT_PROMPT + "\n" + source,
                )
                parsed = safe_json_loads(result)
                if not isinstance(parsed, dict):
                    raise ValueError("Profile response was not a JSON object.")
                st.session_state["profile"] = parsed
                st.session_state["profile_raw"] = source
                st.success("Profile analyzed and stored in this session.")
            except Exception as exc:
                st.error(f"Profile analysis failed: {exc}")

    if st.session_state.get("profile"):
        st.subheader("Current normalized profile")
        st.json(st.session_state["profile"])


# =============================================================================
# TAB 2 — Find positions
# =============================================================================

with tabs[1]:
    st.header("Find Master's / PhD opportunities")

    profile = st.session_state.get("profile", {})

    if not profile:
        st.warning("Build a profile first.")
    else:
        st.subheader("A. AI-powered live discovery")
        st.caption(
            "The live search prototype uses OpenAI's web-search capability. "
            "Results are leads; verify them on the official university page."
        )

        q1, q2, q3 = st.columns(3)
        with q1:
            search_degree = st.selectbox(
                "Degree",
                ["PhD", "Master's", "PhD or Master's"],
                key="search_degree",
            )
        with q2:
            search_countries = st.text_input(
                "Countries / regions",
                value=str(profile.get("target_countries", "")),
                key="search_countries",
            )
        with q3:
            search_funding = st.selectbox(
                "Funding",
                ["Fully funded", "Funded or scholarship", "Any funding"],
                key="search_funding",
            )

        search_keywords = st.text_input(
            "Research keywords",
            value=", ".join(profile.get("research_domains", []))
            if isinstance(profile.get("research_domains"), list)
            else str(profile.get("research_domains", "")),
            key="search_keywords",
        )

        max_results = st.slider("Number of leads", 3, 12, 8)

        st.info(
            "Discovery uses your selected provider's built-in web search. "
            "OpenAI, Gemini, and Anthropic are supported; DeepSeek does not currently "
            "provide native web search."
        )

        if st.button("🔎 Search current opportunities", type="primary"):
            try:
                if not search_keywords.strip():
                    raise ValueError("Enter at least one research keyword/domain.")

                search_prompt = f"""
Use your built-in web-search tool to find up to {max_results} current
{search_degree} opportunities that best match this candidate. Search the live web
before answering, using several focused searches if needed.

Candidate:
{stringify_profile(profile)}

Search constraints:
- Countries/regions: {search_countries}
- Funding: {search_funding}
- Research keywords: {search_keywords}

Rules:
- Prefer official university, research institute, and funder pages.
- Include only opportunities whose source page supports the listed details.
- Exclude opportunities clearly marked closed or with expired deadlines.
- Do not invent deadlines, funding, supervisors, requirements, or URLs.
- Use "Not stated" for details that cannot be verified.
- Include the direct source URL for every opportunity.

Return ONLY valid JSON, without Markdown fences:
{{
  "positions": [
    {{
      "title": "",
      "university": "",
      "country": "",
      "degree": "",
      "funding": "",
      "deadline": "",
      "url": "",
      "description": ""
    }}
  ]
}}
"""

                raw = call_ai(
                    provider=provider,
                    api_key=get_api_key(provider, api_key),
                    model=model,
                    system_prompt=(
                        "You are an academic scholarship discovery assistant. "
                        "Use your built-in live web-search tool to find and verify "
                        "current opportunities. Do not invent facts. Return structured JSON."
                    ),
                    user_prompt=search_prompt,
                    use_web_search=True,
                )
                positions = parse_position_search_response(raw)

                if not positions:
                    raise ValueError("The selected AI provider returned no structured opportunities.")

                st.session_state["positions"] = positions
                st.session_state["search_results_raw"] = raw
                st.success(
                    f"Found {len(positions)} opportunity leads using {provider}'s built-in search."
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")

        st.subheader("B. Add a position manually")
        with st.expander("Add opportunity from URL or pasted advertisement"):
            position_url = st.text_input("Position URL")
            fetch_url = st.button("Fetch page text")

            fetched_text = ""
            if fetch_url and position_url:
                try:
                    fetched_text = extract_webpage_text(position_url)
                    st.session_state["manual_position_text"] = fetched_text
                    st.success("Position page text extracted.")
                except Exception as exc:
                    st.error(f"Could not fetch page: {exc}")

            description_default = st.session_state.get("manual_position_text", "")
            manual_description = st.text_area(
                "Advertisement / position text",
                value=description_default,
                height=220,
            )

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                manual_title = st.text_input("Title")
                manual_university = st.text_input("University / institute")
            with mc2:
                manual_country = st.text_input("Country")
                manual_degree = st.text_input("Degree")
            with mc3:
                manual_funding = st.text_input("Funding")
                manual_deadline = st.text_input("Deadline")

            if st.button("Add this opportunity"):
                if not manual_title and not manual_description:
                    st.error("Provide at least a title or advertisement text.")
                else:
                    st.session_state["positions"].append(
                        position_from_fields(
                            title=manual_title,
                            university=manual_university,
                            country=manual_country,
                            degree=manual_degree,
                            funding=manual_funding,
                            deadline=manual_deadline,
                            url=position_url,
                            description=manual_description,
                        )
                    )
                    st.success("Opportunity added.")

        if st.session_state.get("positions"):
            st.subheader("Saved opportunities")

            rows = []
            for idx, item in enumerate(st.session_state["positions"]):
                rows.append(
                    {
                        "id": idx,
                        "Title": item.get("title", ""),
                        "University": item.get("university", ""),
                        "Country": item.get("country", ""),
                        "Degree": item.get("degree", ""),
                        "Funding": item.get("funding", ""),
                        "Deadline": item.get("deadline", ""),
                        "URL": item.get("url", ""),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("🧠 Analyze all saved opportunities for profile fit"):
                try:
                    compact_positions = [
                        {
                            "title": p.get("title", ""),
                            "university": p.get("university", ""),
                            "country": p.get("country", ""),
                            "degree": p.get("degree", ""),
                            "funding": p.get("funding", ""),
                            "deadline": p.get("deadline", ""),
                            "url": p.get("url", ""),
                            "description": p.get("description", "")[:6000],
                        }
                        for p in st.session_state["positions"]
                    ]

                    result = call_ai(
                        provider=provider,
                        api_key=get_api_key(provider, api_key),
                        model=model,
                        system_prompt=MATCH_SYSTEM,
                        user_prompt=MATCH_PROMPT.format(
                            profile=stringify_profile(profile),
                            positions=json.dumps(
                                compact_positions,
                                indent=2,
                                ensure_ascii=False,
                            ),
                        ),
                    )
                    match_data = safe_json_loads(result)
                    matches = match_data.get("matches", []) if isinstance(match_data, dict) else []

                    if matches:
                        st.session_state["positions"] = matches
                        st.success("Profile-fit analysis completed.")
                    else:
                        st.warning("The model returned no matches.")
                except Exception as exc:
                    st.error(f"Matching failed: {exc}")

            # Show richer results after matching.
            if st.session_state["positions"] and "fit_score" in st.session_state["positions"][0]:
                st.subheader("Fit results")
                fit_rows = []
                for idx, item in enumerate(st.session_state["positions"]):
                    fit_rows.append(
                        {
                            "id": idx,
                            "Title": item.get("title", ""),
                            "University": item.get("university", ""),
                            "Country": item.get("country", ""),
                            "Degree": item.get("degree", ""),
                            "Funding": item.get("funding", ""),
                            "Deadline": item.get("deadline", ""),
                            "Fit score": item.get("fit_score", ""),
                            "URL": item.get("url", ""),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(fit_rows),
                    use_container_width=True,
                    hide_index=True,
                )


# =============================================================================
# TAB 3 — Analyze position
# =============================================================================

with tabs[2]:
    st.header("Analyze a specific position")

    profile = st.session_state.get("profile", {})

    if not profile:
        st.warning("Build a profile first.")
    else:
        saved_positions = st.session_state.get("positions", [])

        options = ["Paste a new advertisement"]
        for i, p in enumerate(saved_positions):
            options.append(
                f"{i}: {p.get('title', 'Untitled')} — {p.get('university', '')}"
            )

        choice = st.selectbox("Position source", options)

        selected = {}
        if choice != "Paste a new advertisement":
            idx = int(choice.split(":", 1)[0])
            selected = saved_positions[idx]
        else:
            title = st.text_input("Position title", key="an_title")
            university = st.text_input("University / institute", key="an_university")
            url = st.text_input("Official position URL", key="an_url")
            description = st.text_area(
                "Advertisement text",
                height=320,
                key="an_description",
            )
            selected = {
                "title": title,
                "university": university,
                "country": "",
                "degree": "",
                "funding": "",
                "deadline": "",
                "url": normalize_url(url),
                "description": description,
            }

        if st.button("🔬 Analyze candidate vs position", type="primary"):
            if not selected.get("description") and not selected.get("url"):
                st.error("Provide the advertisement text or URL.")
            else:
                try:
                    position_text = selected.get("description", "")
                    if not position_text and selected.get("url"):
                        position_text = extract_webpage_text(selected["url"])

                    position_for_prompt = {
                        **selected,
                        "description": position_text[:MAX_POSITION_CHARS],
                    }

                    result = call_ai(
                        provider=provider,
                        api_key=get_api_key(provider, api_key),
                        model=model,
                        system_prompt=POSITION_ANALYSIS_SYSTEM,
                        user_prompt=POSITION_ANALYSIS_PROMPT.format(
                            profile=stringify_profile(profile),
                            position=json.dumps(
                                position_for_prompt,
                                indent=2,
                                ensure_ascii=False,
                            ),
                        ),
                    )

                    parsed = safe_json_loads(result)
                    st.session_state["selected_position"] = position_for_prompt
                    st.session_state["position_analysis"] = parsed
                    st.success("Position analysis completed.")
                except Exception as exc:
                    st.error(f"Position analysis failed: {exc}")

        analysis = st.session_state.get("position_analysis", {})
        if analysis:
            st.subheader("Fit summary")
            st.write(analysis.get("overall_fit_summary", ""))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Eligibility: meets**")
                for item in analysis.get("eligibility", {}).get("meets", []):
                    st.write(f"✅ {item}")
            with c2:
                st.markdown("**Eligibility: not met**")
                for item in analysis.get("eligibility", {}).get("not_met", []):
                    st.write(f"❌ {item}")
            with c3:
                st.markdown("**Eligibility: unknown**")
                for item in analysis.get("eligibility", {}).get("unknown", []):
                    st.write(f"❓ {item}")

            st.subheader("Research fit")
            rf = analysis.get("research_fit", {})
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Strong matches**")
                for item in rf.get("strong_matches", []):
                    st.write(f"✅ {item}")
            with col_b:
                st.markdown("**Gaps / partial matches**")
                for item in rf.get("partial_matches", []) + rf.get("gaps", []):
                    st.write(f"⚠️ {item}")

            st.subheader("Skills fit")
            sf = analysis.get("skills_fit", {})
            st.markdown("**Strong matches**")
            for item in sf.get("strong_matches", []):
                st.write(f"✅ {item}")
            st.markdown("**Missing / weak evidence**")
            for item in sf.get("missing_or_weak", []):
                st.write(f"⚠️ {item}")

            st.subheader("Application strategy")
            for item in analysis.get("application_strategy", []):
                st.write(f"• {item}")

            st.subheader("Documents requested")
            docs = analysis.get("required_documents", [])
            if docs:
                doc_rows = [
                    {
                        "Document": d.get("document", ""),
                        "Required": d.get("required", ""),
                        "Evidence from advertisement": d.get("evidence_from_ad", ""),
                    }
                    for d in docs
                ]
                st.dataframe(
                    pd.DataFrame(doc_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Next steps")
            for item in analysis.get("recommended_next_steps", []):
                st.write(f"➡️ {item}")

            st.subheader("Questions to clarify")
            for item in analysis.get("questions_to_clarify", []):
                st.write(f"❓ {item}")


# =============================================================================
# TAB 4 — Documents
# =============================================================================

with tabs[3]:
    st.header("Prepare application documents")

    profile = st.session_state.get("profile", {})
    position = st.session_state.get("selected_position", {})
    analysis = st.session_state.get("position_analysis", {})

    if not profile:
        st.warning("Build a profile first.")
    elif not position:
        st.warning("Analyze a specific position first.")
    else:
        detected_docs = [
            d.get("document")
            for d in analysis.get("required_documents", [])
            if d.get("document")
        ]

        default_types = [
            "Email to prospective supervisor",
            "Motivation letter",
            "Statement of Interest",
            "Statement of Purpose (SOP)",
            "Research proposal",
            "Personal statement",
            "Custom document",
        ]

        document_type = st.selectbox(
            "Document type",
            default_types,
        )

        if detected_docs:
            st.caption(
                "Documents detected from the advertisement: "
                + ", ".join(detected_docs[:8])
            )

        extra_instructions = st.text_area(
            "Additional instructions",
            placeholder=(
                "For example: 800 words, direct tone, mention thesis, "
                "include 3 research objectives, etc."
            ),
            height=120,
        )

        if document_type == "Email to prospective supervisor":
            extra_instructions = (
                extra_instructions
                + "\nKeep it concise (roughly 150–250 words), specific to the supervisor/position, "
                "with a clear subject line."
            )

        elif document_type == "Research proposal":
            if st.button("🧪 Generate research proposal", type="primary"):
                try:
                    result = call_ai(
                        provider=provider,
                        api_key=get_api_key(provider, api_key),
                        model=model,
                        system_prompt=RESEARCH_PROPOSAL_SYSTEM,
                        user_prompt=DOCUMENT_PROMPT.format(
                            profile=stringify_profile(profile),
                            position=json.dumps(position, indent=2, ensure_ascii=False),
                            analysis=json.dumps(analysis, indent=2, ensure_ascii=False),
                            document_type=document_type,
                            extra_instructions=extra_instructions,
                        ),
                    )
                    st.session_state["generated_document"] = result[:MAX_GENERATED_DOC_CHARS]
                    st.session_state["generated_document_type"] = document_type
                    st.success("Research proposal draft generated.")
                except Exception as exc:
                    st.error(f"Document generation failed: {exc}")
        else:
            if st.button("✍️ Generate document", type="primary"):
                try:
                    result = call_ai(
                        provider=provider,
                        api_key=get_api_key(provider, api_key),
                        model=model,
                        system_prompt=DOCUMENT_SYSTEM,
                        user_prompt=DOCUMENT_PROMPT.format(
                            profile=stringify_profile(profile),
                            position=json.dumps(position, indent=2, ensure_ascii=False),
                            analysis=json.dumps(analysis, indent=2, ensure_ascii=False),
                            document_type=document_type,
                            extra_instructions=extra_instructions,
                        ),
                    )
                    st.session_state["generated_document"] = result[:MAX_GENERATED_DOC_CHARS]
                    st.session_state["generated_document_type"] = document_type
                    st.success("Document draft generated.")
                except Exception as exc:
                    st.error(f"Document generation failed: {exc}")

        generated = st.session_state.get("generated_document", "")
        if generated:
            st.subheader("Editable draft")
            edited = st.text_area(
                "Draft content",
                value=generated,
                height=650,
            )
            st.session_state["generated_document"] = edited

            st.download_button(
                "⬇️ Download Markdown",
                data=edited.encode("utf-8"),
                file_name="scholarship_application_draft.md",
                mime="text/markdown",
            )

            docx_bytes = make_docx(
                st.session_state.get("generated_document_type", "Application Document"),
                edited,
            )

            st.download_button(
                "⬇️ Download DOCX",
                data=docx_bytes,
                file_name="scholarship_application_draft.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption(
    "ScholarshipFit AI MVP · Use official sources to verify deadlines, eligibility, "
    "funding, and submission instructions."
)
