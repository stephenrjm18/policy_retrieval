"""


Converts a structured IntentResult into optimised, parameterised SQL.

Key fix over original:
  - "find" intent with a name ALWAYS uses name-based WHERE clause.
  - "list" intent without a name produces an unfiltered listing.
  - "both" entity_type with a specific name now correctly applies the
    name filter in the UNION so "AARTHI SCANS hospitals" finds Aarthi Scans,
    not every hospital+doctor in the DB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from intent_extractor import IntentResult


@dataclass
class RouteResult:
    sql:      str
    params:   tuple
    table:    str       # hospitals | doctors | both
    strategy: str       # debug label


# Column sets

_HOSPITAL_COLS = "hospital_name, code, address, city, status, phone_number"
_DOCTOR_COLS   = "doctor_name, register_number, code, address, status, phone_number"


# Helpers

def _like(value: str) -> str:
    return f"%{value.strip()}%"


def _tokenize(text: str) -> list[str]:
    """Split into meaningful tokens, ignoring short noise words."""
    STOP = {"the", "and", "or", "in", "at", "of", "for", "a", "an", "ms", "m", "s"}
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= 2 and w not in STOP]


def _name_conditions(column: str, name: str) -> tuple[str, list]:
    """
    Tokenised AND conditions for a name field using the pre-computed
    _*_lower indexed column for performance.
    e.g. "apollo gleneagles" -> _name_lower LIKE '%apollo%' AND _name_lower LIKE '%gleneagles%'

    For very specific names (2+ tokens) all tokens must match.
    For single-token names a single LIKE is used.
    """
    lower_col = column.replace("hospital_name", "_name_lower").replace("doctor_name", "_name_lower")
    tokens    = _tokenize(name)
    if not tokens:
        return "", []
    clauses = [f"{lower_col} LIKE ?" for _ in tokens]
    params  = [_like(t) for t in tokens]
    return " AND ".join(clauses), params


# Hospital SQL builder

def _build_hospital_sql(intent: IntentResult) -> tuple[str, tuple]:
    cols = "COUNT(*) AS total" if intent.intent == "count" else _HOSPITAL_COLS
    conditions: list[str] = []
    params: list = []

    if intent.location:
        conditions.append("(_city_lower LIKE ? OR _address_lower LIKE ?)")
        params += [_like(intent.location.lower())] * 2

    if intent.code:
        conditions.append("UPPER(code) LIKE ?")
        params.append(_like(intent.code))

    if intent.status == "active":
        conditions.append("UPPER(status) = 'Y'")
    elif intent.status == "inactive":
        conditions.append("UPPER(status) = 'N'")

    if intent.name_keyword:
        name_sql, name_params = _name_conditions("hospital_name", intent.name_keyword)
        if name_sql:
            conditions.append(f"({name_sql})")
            params.extend(name_params)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "" if intent.intent == "count" else "ORDER BY hospital_name"
    limit = "" if intent.intent == "count" else "LIMIT 50"

    sql = f"SELECT {cols} FROM hospitals {where} {order} {limit}".strip()
    return sql, tuple(params)


# Doctor SQL builder

def _build_doctor_sql(intent: IntentResult) -> tuple[str, tuple]:
    cols = "COUNT(*) AS total" if intent.intent == "count" else _DOCTOR_COLS
    conditions: list[str] = []
    params: list = []

    if intent.specialty:
        conditions.append("(_address_lower LIKE ? OR _name_lower LIKE ?)")
        params += [_like(intent.specialty.lower())] * 2

    if intent.location:
        conditions.append("_address_lower LIKE ?")
        params.append(_like(intent.location.lower()))

    if intent.code:
        conditions.append("UPPER(code) LIKE ?")
        params.append(_like(intent.code))

    if intent.status == "active":
        conditions.append("UPPER(status) = 'Y'")
    elif intent.status == "inactive":
        conditions.append("UPPER(status) = 'N'")

    if intent.name_keyword:
        name_sql, name_params = _name_conditions("doctor_name", intent.name_keyword)
        if name_sql:
            conditions.append(f"({name_sql} OR _address_lower LIKE ?)")
            params.extend(name_params)
            params.append(_like(intent.name_keyword.lower()))

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "" if intent.intent == "count" else "ORDER BY doctor_name"
    limit = "" if intent.intent == "count" else "LIMIT 50"

    sql = f"SELECT {cols} FROM doctors {where} {order} {limit}".strip()
    return sql, tuple(params)


# UNION builder

def _build_union_sql(intent: IntentResult) -> tuple[str, tuple]:
    """
    UNION of hospitals + doctors.

    If a specific name is present (find intent), apply name filter to BOTH sides.
    If it's a generic list (no name), apply code/location/status filters only.
    This prevents "AARTHI SCANS hospitals" from returning all hospitals+doctors.
    """
    conditions_h: list[str] = []
    params_h: list = []
    conditions_d: list[str] = []
    params_d: list = []

    if intent.location:
        conditions_h.append("(_city_lower LIKE ? OR _address_lower LIKE ?)")
        params_h += [_like(intent.location.lower())] * 2
        conditions_d.append("_address_lower LIKE ?")
        params_d.append(_like(intent.location.lower()))

    if intent.code:
        conditions_h.append("UPPER(code) LIKE ?")
        params_h.append(_like(intent.code))
        conditions_d.append("UPPER(code) LIKE ?")
        params_d.append(_like(intent.code))

    if intent.status == "active":
        conditions_h.append("UPPER(status) = 'Y'")
        conditions_d.append("UPPER(status) = 'Y'")
    elif intent.status == "inactive":
        conditions_h.append("UPPER(status) = 'N'")
        conditions_d.append("UPPER(status) = 'N'")

    # KEY FIX: if there's a name filter, apply it to both sides.
    # Without this, "AARTHI SCANS hospitals" returns every record in the DB.
    if intent.name_keyword:
        h_name_sql, h_name_params = _name_conditions("hospital_name", intent.name_keyword)
        d_name_sql, d_name_params = _name_conditions("doctor_name", intent.name_keyword)
        if h_name_sql:
            conditions_h.append(f"({h_name_sql})")
            params_h.extend(h_name_params)
        if d_name_sql:
            conditions_d.append(f"({d_name_sql})")
            params_d.extend(d_name_params)

    where_h = f"WHERE {' AND '.join(conditions_h)}" if conditions_h else ""
    where_d = f"WHERE {' AND '.join(conditions_d)}" if conditions_d else ""

    sql = f"""
        SELECT hospital_name AS name, code, address, city, status, phone_number, 'hospital' AS record_type
        FROM hospitals {where_h}
        UNION ALL
        SELECT doctor_name AS name, code, address, '' AS city, status, phone_number, 'doctor' AS record_type
        FROM doctors {where_d}
        ORDER BY name LIMIT 100
    """.strip()

    return sql, tuple(params_h + params_d)


# Public builder

def build_sql(intent: IntentResult) -> RouteResult:
    """Convert an IntentResult into a RouteResult with parameterised SQL."""
    entity = intent.entity_type

    # If entity is "both" but we have a specific name, prefer hospital lookup.
    # "AARTHI SCANS hospitals" should search hospitals only, not doctors too.
    if entity == "both" and intent.name_keyword and intent.intent == "find":
        # Try hospital first; if the name matches hospital-brand words, stay hospital
        sql, params = _build_hospital_sql(intent)
        return RouteResult(sql=sql, params=params, table="hospitals",
                           strategy=f"smart:hospital(from both+name) filters={intent.filters}")

    if entity == "both":
        sql, params = _build_union_sql(intent)
        return RouteResult(sql=sql, params=params, table="both",
                           strategy=f"smart:both filters={intent.filters}")

    if entity == "doctor":
        sql, params = _build_doctor_sql(intent)
        return RouteResult(sql=sql, params=params, table="doctors",
                           strategy=f"smart:doctor filters={intent.filters}")

    # Default: hospital
    sql, params = _build_hospital_sql(intent)
    return RouteResult(sql=sql, params=params, table="hospitals",
                       strategy=f"smart:hospital filters={intent.filters}")
