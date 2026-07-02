"""
routers/query_router.py — Semantic query classification using a lightweight LLM.

Uses ROUTER_MODEL (qwen2.5:1.5b by default) to classify every query into one
of five route types. Falls back to heuristic classification if the LLM fails.

Returns strict JSON: {"route": "<ROUTE_TYPE>"}

Valid routes:
  POLICY_QUERY   -> RAG pipeline only
  HOSPITAL_QUERY -> SQL pipeline only
  DOCTOR_QUERY   -> SQL pipeline only
  HYBRID_QUERY   -> both RAG + SQL, results merged
  GENERAL_QUERY  -> direct LLM answer (no retrieval)
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from utils.ai_engine import query_llm
from utils.config import ROUTER_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

RouteType = Literal[
    "POLICY_QUERY",
    "HOSPITAL_QUERY",
    "DOCTOR_QUERY",
    "HYBRID_QUERY",
    "GENERAL_QUERY",
]

_VALID_ROUTES: set[str] = {
    "POLICY_QUERY",
    "HOSPITAL_QUERY",
    "DOCTOR_QUERY",
    "HYBRID_QUERY",
    "GENERAL_QUERY",
}


# LLM-based routing

def _route_via_llm(query: str) -> RouteType | None:
    """Call the lightweight router LLM. Returns None on any failure."""
    try:
        prompt = ROUTER_SYSTEM_PROMPT.format(query=query)
        raw = query_llm(prompt, model=ROUTER_MODEL, temperature=0)
        raw = str(raw).strip()

        # Strip any accidental markdown
        raw_clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("`").strip()
        match = re.search(r"\{.*?\}", raw_clean, re.DOTALL)
        if not match:
            logger.warning("Router: no JSON found in LLM output: %.100s", raw)
            return None

        data  = json.loads(match.group())
        route = str(data.get("route", "")).upper().strip()

        if route in _VALID_ROUTES:
            return route  # type: ignore[return-value]

        logger.warning("Router: invalid route value '%s'", route)
        return None

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Router LLM failed: %s", exc)
        return None


# Heuristic fallback routing
#
# Philosophy: when the LLM fails, err strongly toward POLICY_QUERY.
# Only route to HOSPITAL/DOCTOR when the intent is unambiguously a DB lookup.
# Never route clearly medical/situational questions to GENERAL_QUERY.

_POLICY_SIGNALS = frozenset([
    # Core policy/reimbursement terms
    "policy", "policies", "reimburse", "reimbursement", "claim", "claims",
    "document", "documents", "entitle", "entitlement", "limit", "limits",
    "guideline", "guidelines", "procedure", "procedures", "eligible",
    "eligibility", "referral", "cashless", "preauth",
    "rule", "rules", "regulation", "regulations",
    "coverage", "benefit", "benefits", "apply", "application",
    "form", "forms", "sanction", "approval", "submit", "submission",
    "reimbursable", "advance", "inpatient", "outpatient", "circular",
    # Clinical/admin words common in policy questions
    "discharge", "summary", "sign", "signature", "happens",
    "mandatory", "certificate", "authorization", "authorisation",
    "original", "refused", "available",
    # Situation/scenario words — always POLICY
    "rejected", "reject", "delayed", "delay", "deduction", "salary",
    "forgot", "lost", "missing", "mistake", "wrong", "error",
    "emergency", "dependent", "father", "mother", "parent", "spouse",
    "monthly", "month", "late", "time", "upload", "email", "contact",
    "another", "second", "again", "already", "can", "should", "must",
    "why", "what", "how", "when", "will", "would", "does", "need",
    "medicine", "medicines", "bill", "bills", "receipt", "receipts",
    "code", "codes", "mention", "mentioned",
])

# Only used when clearly no POLICY or HOSPITAL signals present
_QUESTION_WORDS = frozenset(["what", "how", "when", "where", "why", "which"])

_HOSPITAL_SIGNALS = frozenset([
    "hospital", "hospitals", "clinic", "clinics", "healthcare",
    "nursing", "centre", "center", "facility", "facilities",
    "empanelled", "empaneled", "listed",
])

# Strong hospital context — bare name lookups
_HOSPITAL_LIST_SIGNALS = frozenset([
    "list", "show", "find", "search", "near", "around", "location",
    "address", "all", "available", "display",
])

_DOCTOR_SIGNALS = frozenset([
    "dr", "doc",
    "doctor", "doctors", "physician", "physicians", "specialist",
    "specialists", "surgeon", "surgeons", "consultant", "consultants",
    "cardiologist", "neurologist", "orthopedic", "gynaecologist",
    "gynecologist", "dentist", "psychiatrist", "dermatologist",
    "oncologist", "radiologist", "urologist", "ophthalmologist",
    "paediatrician", "gastroenterologist", "nephrologist",
    "pulmonologist", "endocrinologist", "rheumatologist", "ent",
])

_GREET_SIGNALS = frozenset([
    "hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "good", "morning",
    "afternoon", "evening",
])

# Phrases that override HOSPITAL/DOCTOR routing -> force POLICY_QUERY
# These indicate a situational/procedural question, not a DB lookup
_POLICY_OVERRIDE_PHRASES = [
    r"\bdidn.t mention\b",
    r"\bwhat should i do\b",
    r"\bwhat do i do\b",
    r"\bhow do i\b",
    r"\bcan i\b",
    r"\bwill .* happen\b",
    r"\bwhy .* my\b",
    r"\bmy .* didn\b",
    r"\bmy .* got\b",
    r"\bmy .* rejected\b",
    r"\bmy .* delayed\b",
    r"\bi (forgot|lost|submitted|took|already)\b",
    r"\bwhat happens\b",
    r"\bshould i\b",
    r"\bdo i need\b",
    r"\bis .* mandatory\b",
    r"\bis .* compulsory\b",
    r"\bis .* required\b",
]

import re as _re_fb
_POLICY_OVERRIDE_RE = _re_fb.compile(
    "|".join(_POLICY_OVERRIDE_PHRASES),
    _re_fb.IGNORECASE,
)


def _heuristic_route(query: str) -> RouteType:
    """
    Keyword-based fallback routing.
    Strong bias toward POLICY_QUERY — only routes to HOSPITAL/DOCTOR
    when the query is clearly a database lookup with no situational language.
    """
    q     = query.lower()
    words = set(_re_fb.findall(r"[a-z]+", q))

    # If any policy-override phrase is present, always route to POLICY
    if _POLICY_OVERRIDE_RE.search(q):
        return "POLICY_QUERY"

    def _match(signals: frozenset, wordset: set) -> bool:
        for w in wordset:
            for s in signals:
                if w == s or (len(s) > 5 and w.startswith(s)):
                    return True
        return False

    q_lower    = query.strip().lower()
    first_word = q_lower.split()[0] if q_lower.split() else ""

    has_policy   = _match(_POLICY_SIGNALS, words)
    has_hospital = _match(_HOSPITAL_SIGNALS, words)
    has_hosp_list = bool(_HOSPITAL_LIST_SIGNALS & words)
    _doc_words   = words.copy()
    if "dr" in _doc_words and first_word != "dr":
        _doc_words.discard("dr")
    has_doctor   = _match(_DOCTOR_SIGNALS, _doc_words)
    has_greet    = bool(_GREET_SIGNALS & words) and len(words) < 5

    # DR. at start of query -> doctor lookup even in heuristic mode
    if _re_fb.match(r'^dr[\.\s]', q_lower):
        return "DOCTOR_QUERY"

    # M/S. prefix or bare facility name -> hospital lookup
    if _re_fb.match(r'^m/s[\.\s]', q_lower):
        return "HOSPITAL_QUERY"

    # Pure greeting with no medical context
    if has_greet and not has_policy and not has_hospital and not has_doctor:
        return "GENERAL_QUERY"

    # Both policy and hospital/doctor signals -> hybrid
    if has_policy and (has_hospital or has_doctor):
        return "HYBRID_QUERY"

    # Doctor lookup: only if no policy signals and dr at start or clear doctor intent
    if has_doctor and not has_policy:
        return "DOCTOR_QUERY"

    # Hospital lookup: only if no policy signals AND there's a list/find intent
    # (prevents "my hospital didn't..." from routing to HOSPITAL_QUERY)
    if has_hospital and not has_policy and has_hosp_list:
        return "HOSPITAL_QUERY"

    # Facility name lookup (bare name, no policy language): treat as hospital query
    # e.g. "M/S. NEUBERG EHRLICH LABORATORY" — no verbs, just a name
    if has_hospital and not has_policy and not _QUESTION_WORDS & words:
        return "HOSPITAL_QUERY"

    # Default: everything else is POLICY (medical domain questions)
    # This is the key change — we never fall through to GENERAL_QUERY
    # for anything that could remotely be a medical policy question
    if has_policy or _QUESTION_WORDS & words:
        return "POLICY_QUERY"

    # Last resort: if it mentions medical/CGHS keywords, treat as policy
    _MEDICAL_TERMS = frozenset([
        "cghs", "hvf", "medical", "health", "treatment", "hospital",
        "advance", "bill", "claim", "reimburse",
    ])
    if _MEDICAL_TERMS & words:
        return "POLICY_QUERY"

    return "GENERAL_QUERY"


# Public API

import re as _re

# Pre-compiled patterns for instant pre-routing (no LLM call needed)
_DR_START_RE  = _re.compile(r'^dr[\.\s]', _re.IGNORECASE)
_DR_ACTION_RE = _re.compile(
    r'\b(find|get|show|list|give|fetch|search|lookup|tell me about|what is|who is)\s+(me\s+)?dr[\.\s]',
    _re.IGNORECASE
)

_POLICY_WORDS_RE = _re.compile(
    r'\b(what happens|if a|if the|when a|when the|should|must|does|refuse|sign|discharge|'
    r'summary|procedure|rule|policy|claim|document|reimburse|guideline|circular|entitle|'
    r'allowed|permitted|required|mandatory|rejected|salary|deduction|forgot|lost|missing|'
    r'didn.t|can i|do i need|why|how do|what do|should i|will it|will my|my .* got|'
    r'another|already|monthly|email|upload|contact|dependent|emergency|advance)\b',
    _re.IGNORECASE
)

_HOSPITAL_BRAND_RE = _re.compile(
    r'\b(hospital|hospitals|clinic|clinics|care|centre|center|memorial|'
    r'healthcare|health care|eye|dental|specialities|specialties|vision|'
    r'nursing|institute|foundation|maternity|diagnostic|laboratory|labs|'
    r'imaging|pharmacy|orthopaedic|multispeciality|multispecialty|polyclinic|'
    r'laboratory|lab|scan|scans|m\/s)\b',
    _re.IGNORECASE
)

# Situational/procedural overrides — these are POLICY even if CGHS/hospital mentioned
_SITUATIONAL_RE = _re.compile(
    r"\b(didn.t|did not|forgot|lost|missing|rejected|delayed|"
    r"what should|how should|can i|should i|will it|why did|why was|"
    r"my hospital|my bill|my claim|my advance|my treatment|already took|"
    r"submitted late|no mention|not mentioned|contact id|upload time|"
    r"approximate time|how long|how much time)\b",
    _re.IGNORECASE
)

# Direct-POLICY fast precheck
# These are short queries whose words are unambiguously in the HVF/CGHS policy
# domain. Skip the LLM entirely — route straight to POLICY_QUERY.
# Rule: ALL words must be from this set (after stripping noise words).
# This catches: "email id for online submission", "upload documents time",
# "contact email", "submission email id", "advance email", etc.
_DIRECT_POLICY_KEYWORDS = frozenset([
    "email", "submission", "submit", "online", "upload", "document", "documents",
    "advance", "claim", "claims", "reimbursement", "reimburse", "medical",
    "bill", "bills", "salary", "deduction", "certificate", "discharge",
    "referral", "emergency", "dependent", "medicine", "medicines", "receipt",
    "receipts", "approval", "sanction", "mandatory", "required", "procedure",
    "contact", "address", "id", "number", "phone", "policy", "guideline",
    "circular", "entitlement", "eligibility", "cashless", "preauth",
    "inpatient", "outpatient", "limit", "maximum", "minimum", "amount",
    "hvf", "cghs", "ama", "csma", "esi", "echs",
])
_DIRECT_POLICY_NOISE = frozenset([
    "for", "the", "a", "an", "of", "to", "in", "at", "on", "by", "with",
    "and", "or", "is", "are", "was", "were", "be", "been", "get", "give",
    "me", "my", "i", "we", "you", "what", "where", "when", "how", "which",
    "please", "need", "want", "tell", "show", "find", "know",
])


def _is_direct_policy(query: str) -> bool:
    """
    Returns True when every non-noise word in the query is a known
    CGHS/HVF policy-domain keyword — meaning the LLM is not needed.
    Minimum 1 non-noise word must match to avoid vacuous matches.
    """
    words = [w for w in _re.findall(r"[a-z]+", query.lower())
             if w not in _DIRECT_POLICY_NOISE]
    if not words:
        return False
    return all(w in _DIRECT_POLICY_KEYWORDS for w in words)


def _fast_precheck(query: str):
    """
    Instant route for unambiguous patterns before calling the LLM.
    Returns a RouteType string, or None to let the LLM decide.

    Priority order:
      1. Direct-policy keyword match  -> POLICY_QUERY  (skip LLM)
      2. Situational/procedural lang  -> None (let LLM decide, it's clearly POLICY)
      3. Policy/procedure language    -> None (let LLM decide)
      4. Hospital brand present       -> None (let LLM decide)
      5. DR. at start / after verb    -> DOCTOR_QUERY
    """
    q = query.strip()

    # 1. Short queries made entirely of policy-domain words -> skip LLM
    if _is_direct_policy(q):
        return "POLICY_QUERY"

    # 2-3. Queries with situational or policy phrasing -> let LLM try first
    if _SITUATIONAL_RE.search(q):
        return None
    if _POLICY_WORDS_RE.search(q):
        return None

    # 4. Hospital/facility brand words present -> let LLM decide entity type
    if _HOSPITAL_BRAND_RE.search(q):
        return None

    # 5. DR. at the very start: "DR.KALAVATHI M details"
    if _DR_START_RE.match(q):
        return "DOCTOR_QUERY"

    # DR. after an action verb: "find DR. Priya"
    if _DR_ACTION_RE.search(q):
        return "DOCTOR_QUERY"

    return None


def classify_query(query: str) -> RouteType:
    """
    Classify query into a route type.
    Order: fast pre-check -> LLM -> sanity-check override -> heuristic fallback.
    Logs timing and decision.

    Key intelligence: if the LLM returns GENERAL_QUERY but our heuristic
    (which knows the HVF domain deeply) says otherwise, we trust the heuristic.
    The small router LLM (1.5B) frequently misclassifies short domain-specific
    queries like "email id for online submission" as general conversation.
    """
    t0 = time.perf_counter()

    # 1. Instant pre-check for obvious patterns (no LLM needed)
    precheck = _fast_precheck(query)
    if precheck:
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info("Router: route=%s  method=precheck  elapsed=%dms  query='%.80s'",
                    precheck, elapsed, query)
        return precheck

    # 2. LLM-based semantic classification
    route = _route_via_llm(query)
    method = "llm"

    # 3. Heuristic fallback if LLM hard-failed (returned None)
    if route is None:
        route  = _heuristic_route(query)
        method = "heuristic"

    # 4. Sanity-check override: if the LLM returned GENERAL_QUERY, verify
    #    it with the heuristic. The small LLM often calls domain-specific
    #    short queries (email, upload, submission, advance) as GENERAL_QUERY.
    #    The heuristic is conservative and domain-aware — if it disagrees, it wins.
    elif route == "GENERAL_QUERY":
        heuristic_route = _heuristic_route(query)
        if heuristic_route != "GENERAL_QUERY":
            logger.info(
                "Router: LLM said GENERAL_QUERY but heuristic says %s — overriding.  query='%.80s'",
                heuristic_route, query,
            )
            route  = heuristic_route
            method = "llm+heuristic_override"

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Router: route=%s  method=%s  elapsed=%dms  query='%.80s'",
        route, method, elapsed, query,
    )
    return route
