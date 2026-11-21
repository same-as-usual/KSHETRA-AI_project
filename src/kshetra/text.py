"""Name normalisation helpers used to join datasets that spell districts differently."""

from __future__ import annotations

import re

# Old or alternate state names -> canonical spelling used across KSHETRA-AI.
STATE_ALIASES = {
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
    "MAHARASTRA": "MAHARASHTRA",
    "DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "DADRA AND NAGAR HAVELI": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN AND NICOBAR ISLANDS",
}

_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_SPACES = re.compile(r"\s+")


def norm_name(value: object) -> str:
    """Upper-case, replace '&' with 'AND', strip punctuation and collapse spaces.

    >>> norm_name("Y.S.R.  ")
    'YSR'
    >>> norm_name("Andaman & Nicobar Islands")
    'ANDAMAN AND NICOBAR ISLANDS'
    """
    if value is None:
        return ""
    text = str(value).upper().replace(".", "").replace("&", " AND ")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def norm_state(value: object) -> str:
    """Normalise a state name and map known historical aliases."""
    name = norm_name(value)
    return STATE_ALIASES.get(name, name)


def parse_number(value: object) -> float:
    """Parse NFHS-style cells such as '(29.5)', '*', '1,234', '27.12%' or ' 82.7 '.

    Values in parentheses are small-sample estimates; they are kept but the
    caller can flag them. Suppressed cells ('*') and blanks return NaN.

    >>> parse_number("(29.5)")
    29.5
    >>> parse_number("*")
    nan
    >>> parse_number("27.12%")
    27.12
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "*", "NA", "N/A", "-"}:
        return float("nan")
    text = text.strip("()").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")
