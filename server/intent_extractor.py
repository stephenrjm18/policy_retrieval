"""
intent_extractor.py — LLM-powered intent understanding for SQL queries.

Extracts structured intent from user queries, which is then passed to
smart_sql_builder.py to produce parameterised SQL — no second LLM call needed.

Uses the fast lightweight INTENT_MODEL (default: qwen2.5:1.5b).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from utils.ai_engine import query_llm
from utils.config import INTENT_MODEL, OLLAMA_URL
from utils.logger import get_logger

logger = get_logger(__name__)

# Intent structure

@dataclass
class IntentResult:
    """Structured representation of the user's query intent."""

    entity_type:       str               # hospital | doctor | both | unknown
    intent:            str               # find | list | contact | check_status | count | general
    filters:           dict = field(default_factory=dict)
    needs_sql:         bool = True
    fallback_text:     Optional[str] = None
    raw_llm_output:    Optional[str] = None
    extraction_method: str = "llm"      # llm | heuristic

    @property
    def location(self) -> Optional[str]:
        return self.filters.get("location")

    @property
    def name_keyword(self) -> Optional[str]:
        return self.filters.get("name")

    @property
    def specialty(self) -> Optional[str]:
        return self.filters.get("specialty")

    @property
    def code(self) -> Optional[str]:
        return self.filters.get("code")

    @property
    def status(self) -> Optional[str]:
        return self.filters.get("status")

    @property
    def phone_requested(self) -> bool:
        return bool(self.filters.get("phone_requested", False))


# Intent extraction prompt

_INTENT_PROMPT = """\
You are an intent extractor for a hospital and doctor directory.
The database has:
  - hospitals: hospital_name, code (CGHS/AMA/CSMA/ESI/ECHS), address, city, status (Y=active, N=inactive), phone_number
  - doctors: doctor_name, register_number, code, address, status, phone_number

Analyze the query and return EXACTLY this JSON (no markdown):

{{
  "entity_type": "<hospital | doctor | both | unknown>",
  "intent": "<find | list | contact | check_status | count | general>",
  "filters": {{
    "location": "<city/area or null>",
    "name": "<hospital or doctor name keyword or null>",
    "specialty": "<medical specialty if mentioned or null>",
    "code": "<CGHS | AMA | CSMA | ESI | ECHS | null>",
    "status": "<active | inactive | null>",
    "phone_requested": <true | false>
  }},
  "needs_sql": <true | false>,
  "fallback_text": "<direct answer for trivial questions, else null>"
}}

Rules:
- "both" entity_type for queries spanning hospitals AND doctors
- "general" intent for greetings / off-topic (set needs_sql=false)
- Extract location from: "near X", "in X", "around X", "at X"
- Map medical terms: cardiologist->cardiology, ortho->orthopedics, neuro->neurology, gynaec->gynaecology
- phone_requested=true if user asks for phone/contact/number
- "active" status if user says "active", "working", "empanelled", "currently"
- If the query contains a SPECIFIC facility name (e.g. "SHIFA HOSPITALS details",
  "SRM MEDICAL COLLEGE HOSPITAL", "TRINITY ACUTE CARE HOSPITAL", "NEUBERG EHRLICH LABORATORY",
  "AARTHI SCANS"), set intent="find" and name=<the facility name>, NOT intent="list".
  intent="list" is ONLY for generic list requests like "show all CGHS hospitals" or "hospital list".
- For bare facility name lookups (query is just a name like "M/S. NEUBERG EHRLICH LABORATORY"),
  entity_type="hospital" and intent="find" with the name in filters.
- For "AARTHI SCANS hospitals" — name="AARTHI SCANS", entity_type="hospital", intent="find",
  do NOT set entity_type="both" just because it mentions a facility name with hospital.
- Return ONLY the JSON, no prose.

User query: {query}
JSON:"""


# LLM-based extraction

def _extract_via_llm(query: str) -> Optional[IntentResult]:
    """Call the intent LLM. Returns None on any failure."""
    try:
        prompt = _INTENT_PROMPT.format(query=query)
        raw = query_llm(prompt, model=INTENT_MODEL, temperature=0)
        raw = str(raw).strip()

        raw_clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
        match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        if not match:
            logger.warning("IntentExtractor: no JSON in LLM output: %s", raw[:150])
            return None

        data = json.loads(match.group())

        entity_type = data.get("entity_type", "unknown")
        intent      = data.get("intent", "find")
        filters     = data.get("filters", {})
        needs_sql   = data.get("needs_sql", True)
        fallback    = data.get("fallback_text")

        clean_filters = {
            k: v for k, v in filters.items()
            if v is not None and v != "" and v != "null"
        }

        return IntentResult(
            entity_type       = entity_type,
            intent            = intent,
            filters           = clean_filters,
            needs_sql         = bool(needs_sql),
            fallback_text     = fallback if fallback and fallback != "null" else None,
            raw_llm_output    = raw,
            extraction_method = "llm",
        )

    except json.JSONDecodeError as exc:
        logger.warning("IntentExtractor: JSON parse error: %s", exc)
        return None
    except Exception as exc:
        logger.warning("IntentExtractor: LLM call failed: %s", exc)
        return None


# Heuristic fallback

_DOCTOR_WORDS = frozenset([
    "doctor", "doctors", "dr", "physician", "physicians", "specialist",
    "specialists", "surgeon", "surgeons", "consultant", "consultants",
    "cardiologist", "neurologist", "orthopedic", "gynaecologist",
    "gynaecologist", "dentist", "psychiatrist", "dermatologist",
    "paediatrician", "radiologist", "urologist", "ophthalmologist",
    "ent", "oncologist", "gastroenterologist", "nephrologist",
    "pulmonologist", "endocrinologist", "rheumatologist",
])

_HOSPITAL_WORDS = frozenset([
    "hospital", "hospitals", "clinic", "clinics", "medical", "center",
    "medical centre", "healthcare", "nursing home", "centre",
    "care", "memorial", "vision", "eye", "dental", "specialities",
    "specialties", "health", "institute", "foundation", "maternity",
    "diagnostic", "laboratory", "labs", "imaging", "pharmacy",
    "orthopaedic", "multispeciality", "multispecialty", "polyclinic",
    "scans", "scan", "lab",
])

# Generic list intent signals — bare name lookups should NOT trigger "list"
_LIST_INTENT_WORDS = frozenset([
    "list", "all", "show all", "display all", "every", "give me all",
])

_CODE_MAP = {
    "cghs": "CGHS", "ama": "AMA", "csma": "CSMA",
    "esi":  "ESI",  "echs": "ECHS",
}

_LOCATION_PREPS = ("in ", "at ", "near ", "from ", "around ")

_SPECIALTY_MAP = {
    "cardio": "cardiology",    "cardiologist": "cardiology",
    "ortho":  "orthopedics",   "orthopedic": "orthopedics",
    "neuro":  "neurology",     "neurologist": "neurology",
    "gynaec": "gynaecology",   "gynaecologist": "gynaecology",
    "gynae":  "gynaecology",   "gynecologist": "gynaecology",
    "ent":    "ENT",
    "onco":   "oncology",      "oncologist": "oncology",
    "derm":   "dermatology",   "dermatologist": "dermatology",
    "paedia": "paediatrics",   "paediatrician": "paediatrics",
    "dental": "dentistry",     "dentist": "dentistry",
    "psychi": "psychiatry",    "psychiatrist": "psychiatry",
    "radio":  "radiology",     "radiologist": "radiology",
    "urology": "urology",      "urologist": "urology",
    "ophthal": "ophthalmology","ophthalmologist": "ophthalmology",
    "gastro": "gastroenterology", "gastroenterologist": "gastroenterology",
    "nephro": "nephrology",    "nephrologist": "nephrology",
    "pulmon": "pulmonology",   "pulmonologist": "pulmonology",
    "endocrin": "endocrinology", "endocrinologist": "endocrinology",
    "rheumat": "rheumatology", "rheumatologist": "rheumatology",
}


def _heuristic_extract(query: str) -> IntentResult:
    """Rule-based fallback. Always returns a result."""
    q     = query.lower()
    words = set(re.findall(r"[a-z]+", q))

    # "dr" alone is a weak doctor signal
    _doc_words_h = words.copy()
    _strong_hospital = _HOSPITAL_WORDS & words
    if "dr" in _doc_words_h and _strong_hospital:
        _doc_words_h.discard("dr")

    has_doctor   = bool(_DOCTOR_WORDS & _doc_words_h)
    has_hospital = bool(_HOSPITAL_WORDS & words)

    if has_doctor and has_hospital:
        entity_type = "both"
    elif has_doctor:
        entity_type = "doctor"
    elif has_hospital:
        entity_type = "hospital"
    else:
        # Bare name lookup with no clear entity type — treat as hospital
        entity_type = "hospital"

    # Determine intent: only use "list" if generic list words AND no specific name signals
    # For specific named-facility queries, always use "find"
    q_words = set(re.findall(r"[a-z]+", q))
    has_list_intent = bool(_LIST_INTENT_WORDS & q_words)

    # Check for specific name (query is more than just generic words)
    _generic_stop = {"list", "show", "all", "display", "fetch", "get", "hospital",
                     "hospitals", "clinic", "clinics", "details", "info", "active",
                     "cghs", "ama", "csma", "esi", "echs", "find", "give", "me",
                     "please", "near", "in", "at", "from", "around", "and", "or",
                     "with", "for", "the", "a", "an"}
    remaining_words = [w for w in re.findall(r"[a-z]+", q) if w not in _generic_stop and len(w) > 2]

    has_specific_name = bool(remaining_words) and not has_list_intent

    if any(w in words for w in ("phone", "contact", "number", "call")):
        intent = "contact"
    elif any(w in words for w in ("status", "active", "inactive", "working")):
        intent = "check_status"
    elif any(w in words for w in ("count", "many", "total")):
        intent = "count"
    elif any(w in words for w in ("hi", "hello", "hey", "thanks", "thank")):
        intent = "general"
    elif has_list_intent and not has_specific_name:
        intent = "list"
    else:
        # Default to find — safer than list for named facilities
        intent = "find"

    filters: dict = {}

    for prep in _LOCATION_PREPS:
        idx = q.find(prep)
        if idx != -1:
            after = query[idx + len(prep):].strip()
            city  = re.split(r"[,?.]|\band\b|\bor\b|\bwith\b", after, maxsplit=1)[0].strip()
            if city and len(city) > 1:
                filters["location"] = city
                break

    for token, code in _CODE_MAP.items():
        if token in words:
            filters["code"] = code
            break

    for token, spec in _SPECIALTY_MAP.items():
        if token in q:
            filters["specialty"] = spec
            break

    if any(w in words for w in ("active", "empanelled", "working", "currently")):
        filters["status"] = "active"
    elif any(w in words for w in ("inactive", "closed")):
        filters["status"] = "inactive"

    if any(w in words for w in ("phone", "contact", "number", "call")):
        filters["phone_requested"] = True

    # Extract name from query — use the full meaningful portion
    stop = _DOCTOR_WORDS | _HOSPITAL_WORDS | {
        "the", "a", "an", "is", "are", "what", "which", "where", "how",
        "many", "give", "me", "please", "contact", "number", "phone",
        "name", "address", "details", "info", "all", "any", "their",
        "its", "list", "show", "find", "get", "near", "in", "at",
        "from", "around", "and", "or", "with", "for", "ms", "m",
    } | set(_CODE_MAP.keys())

    # For facility name lookups, extract multi-word name
    # Try to get the original casing for the name
    orig_words = re.findall(r"[A-Za-z0-9/\.]+", query)
    name_words = [w for w in orig_words if w.lower() not in stop and len(w) > 1]
    if name_words:
        filters["name"] = " ".join(name_words[:4])  # up to 4 words

    needs_sql = intent != "general"

    return IntentResult(
        entity_type       = entity_type,
        intent            = intent,
        filters           = filters,
        needs_sql         = needs_sql,
        fallback_text     = (
            "Hello! I can help you find hospitals and doctors. What are you looking for?"
            if intent == "general" else None
        ),
        extraction_method = "heuristic",
    )


# Public API

def extract_intent(query: str) -> IntentResult:
    """
    Main entry point.
    Tries LLM extraction first; falls back to heuristic if LLM fails.
    """
    t0 = time.perf_counter()
    result = _extract_via_llm(query)
    elapsed = (time.perf_counter() - t0) * 1000

    if result is None:
        logger.info("IntentExtractor: LLM failed (%.0fms) — using heuristic", elapsed)
        result = _heuristic_extract(query)
    else:
        logger.info(
            "IntentExtractor: LLM (%.0fms) entity=%s intent=%s filters=%s",
            elapsed, result.entity_type, result.intent, result.filters,
        )

    return result
