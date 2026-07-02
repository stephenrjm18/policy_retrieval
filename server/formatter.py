"""
formatter.py — Direct Result Formatter.

Converts raw database rows into readable text without any LLM call.
Pure Python — fast.
"""

from __future__ import annotations

from typing import Any, Optional

def _status_label(val: Any) -> str:
    v = str(val).strip().upper() if val else ""
    if v == "Y":
        return "✅ Active"
    if v in ("N", ""):
        return "❌ Inactive"
    return str(val)

def _safe(val: Any, default: str = "") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return default if s.lower() == "nan" else s

def _fmt_hospital(row: dict, idx: int) -> str:
    name   = _safe(row.get("hospital_name"), "—") or "—"
    code   = _safe(row.get("code"), "—") or "—"
    addr   = _safe(row.get("address"), "—") or "—"
    city   = _safe(row.get("city"), "")
    phone  = _safe(row.get("phone_number"), "")
    status = _status_label(row.get("status", ""))

    parts = [f"**{idx}. {name}**"]
    parts.append(f"   Code    : {code}")
    parts.append(f"   Address : {addr}" + (f", {city}" if city and city not in addr else ""))
    if phone and phone not in ("nan", ""):
        parts.append(f"   Phone   : {phone}")
    parts.append(f"   Status  : {status}")
    return "\n".join(parts)

def _fmt_doctor(row: dict, idx: int) -> str:
    name   = _safe(row.get("doctor_name"), "—") or "—"
    reg    = _safe(row.get("register_number"), "")
    code   = _safe(row.get("code"), "—") or "—"
    addr   = _safe(row.get("address"), "—") or "—"
    phone  = _safe(row.get("phone_number"), "")
    status = _status_label(row.get("status", ""))

    parts = [f"**{idx}. {name}**"]
    if reg and reg not in ("nan", ""):
        parts.append(f"   Reg No  : {reg}")
    parts.append(f"   Code    : {code}")
    parts.append(f"   Address : {addr}")
    if phone and phone not in ("nan", ""):
        parts.append(f"   Phone   : {phone}")
    parts.append(f"   Status  : {status}")
    return "\n".join(parts)

def _fmt_mixed(row: dict, idx: int) -> str:
    record_type = _safe(row.get("record_type"), "record")
    name        = _safe(row.get("name"), "—") or "—"
    code        = _safe(row.get("code"), "—") or "—"
    addr        = _safe(row.get("address"), "—") or "—"
    city        = _safe(row.get("city"), "")
    phone       = _safe(row.get("phone_number"), "")
    status      = _status_label(row.get("status", ""))
    emoji       = "🏥" if record_type == "hospital" else "👨‍⚕️"

    parts = [f"**{idx}. {emoji} {name}** _{record_type.title()}_"]
    parts.append(f"   Code    : {code}")
    parts.append(f"   Address : {addr}" + (f", {city}" if city and city not in addr else ""))
    if phone and phone not in ("nan", ""):
        parts.append(f"   Phone   : {phone}")
    parts.append(f"   Status  : {status}")
    return "\n".join(parts)

def _fmt_generic(row: dict, idx: int) -> str:
    lines = [f"**{idx}.**"]
    for k, v in row.items():
        if k.startswith("_"):
            continue  # skip internal _*_lower columns
        if v is not None and str(v).strip() not in ("", "nan"):
            lines.append(f"   {k.replace('_', ' ').title()}: {v}")
    return "\n".join(lines)

def format_rows(
    rows: list[dict], table: str, user_query: str, intent=None
) -> str:
    """
    Convert database rows into human-readable markdown text.
    No LLM call. Pure Python.
    """
    # Strip internal columns from all rows
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    # COUNT query result
    if rows and "total" in rows[0]:
        total  = rows[0]["total"]
        entity = "hospitals" if table != "doctors" else "doctors"
        return f"There are **{total} {entity}** matching your query."

    # Empty result
    if not rows:
        entity = {"hospitals": "hospitals", "doctors": "doctors"}.get(table, "records")
        suggestion = ""
        if intent and intent.filters:
            applied = []
            if intent.location:   applied.append(f"location '{intent.location}'")
            if intent.specialty:  applied.append(f"specialty '{intent.specialty}'")
            if intent.code:       applied.append(f"code '{intent.code}'")
            if intent.status:     applied.append(f"status '{intent.status}'")
            if applied:
                suggestion = f"\n\nFilters applied: {', '.join(applied)}."
                suggestion += "\nTry removing one filter or broadening your search."
        return (
            f"No {entity} found matching your query.{suggestion}\n\n"
            f"Try: 'list all {entity}' or broaden your search terms."
        )

    # Mixed UNION result
    if table == "both" or (rows and "record_type" in rows[0]):
        total  = len(rows)
        header = f"Found **{total} result{'s' if total != 1 else ''}** (hospitals & doctors):\n"
        header += "─" * 40
        body   = "\n\n".join(_fmt_mixed(row, i) for i, row in enumerate(rows, 1))
        footer = "\n\n_(Showing first 100 results)_" if total >= 100 else ""
        return f"{header}\n\n{body}{footer}"

    # Standard single-table result
    is_doctors   = (table == "doctors") or (rows and "doctor_name" in rows[0])
    entity_label = "Doctor" if is_doctors else "Hospital"
    total        = len(rows)
    header       = f"Found **{total} {entity_label}{'s' if total != 1 else ''}**:\n" + "─" * 40

    formatted = []
    for i, row in enumerate(rows, 1):
        if is_doctors:
            formatted.append(_fmt_doctor(row, i))
        elif "hospital_name" in row:
            formatted.append(_fmt_hospital(row, i))
        else:
            formatted.append(_fmt_generic(row, i))

    body   = "\n\n".join(formatted)
    footer = "\n\n_(Showing first 50 results — refine your query for fewer results)_" if total == 50 else ""
    return f"{header}\n\n{body}{footer}"

def format_error(message: str) -> str:
    return f"⚠️ {message}\n\nPlease try rephrasing your query."

def format_sql_error(sql: str, exc: Exception) -> str:
    return (
        f"⚠️ Could not retrieve results.\n\n"
        f"Error: {exc}\n\n"
        f"Please try a simpler query."
    )
