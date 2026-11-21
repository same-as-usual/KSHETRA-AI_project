"""PRAGATI assistant: local retrieval + constrained, evidence-grounded answer templates.

No external LLM, API key, or network call is used. Responses are generated only from the
local district profile and the scheme-proxy definitions; unsupported questions are declined.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .schemes import SCHEME_PROXIES, scheme_gap_frame
from .text import norm_name, norm_state

_DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def _district_row(profiles: pd.DataFrame, district: str | None, question: str, state: str | None = None) -> pd.Series | None:
    if district:
        wanted = norm_name(district)
        rows = profiles[profiles["district_key"].eq(wanted)]
        if state and len(rows) > 1:
            rows = rows[rows["state_key"].eq(norm_state(state))]
        if len(rows) == 1:
            return rows.iloc[0]
    q = norm_name(question)
    matches = profiles[profiles["district_key"].map(lambda n: bool(n and n in q))]
    if len(matches) == 1:
        return matches.iloc[0]
    if len(matches) > 1:
        return matches.loc[matches["district_key"].str.len().idxmax()]
    return None


def _is_hindi(question: str) -> bool:
    return bool(_DEVANAGARI.search(question))


def retrieve_evidence(question: str, profiles: pd.DataFrame, district: str | None = None,
                      top_k: int = 6, state: str | None = None) -> list[dict[str, Any]]:
    """Retrieve exact district facts or keyword-overlap records for a local evidence panel."""
    row = _district_row(profiles, district, question, state)
    evidence: list[dict[str, Any]] = []
    if row is not None:
        evidence.append({
            "source": "NFHS-5 district profile + KSHETRA-AI composite",
            "district": f"{row.get('district_raw','')} / {row.get('state_raw','')}",
            "fact": (f"percentile={row.get('composite_percentile', float('nan')):.1f}; "
                     f"tier={row.get('risk_tier')}; main_domain={row.get('primary_domain')}; "
                     f"top_drivers={row.get('top_drivers')}; confidence={row.get('confidence')}"),
        })
        if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
            evidence.append({
                "source": "Census 2011 MPI candidate table (provenance to confirm)",
                "district": f"{row.get('district_raw','')} / {row.get('state_raw','')}",
                "fact": f"MPI HCR={float(row['mpi_hcr_pct']):.1f}% ({row.get('mpi_match')} name match)",
            })
        gaps = scheme_gap_frame(pd.DataFrame([row]))
        for _, gap in gaps.dropna(subset=["gap_proxy_pct"]).iterrows():
            evidence.append({
                "source": f"NFHS-5 (2019-21) scheme service-coverage proxy; {gap['scheme']}",
                "district": f"{row.get('district_raw','')} / {row.get('state_raw','')}",
                "fact": (f"coverage proxy={gap['coverage_proxy_pct']:.1f}%; "
                         f"uncovered proxy={gap['gap_proxy_pct']:.1f}%; not scheme enrollment"),
            })
        return evidence[:top_k]

    # No exact district found: retrieve a few matching profile records by term overlap.
    terms = {t for t in re.findall(r"[A-Z0-9]{3,}", norm_name(question))}
    if not terms:
        return []
    candidates = profiles.copy()
    candidates["_text"] = (candidates["state_raw"].astype(str) + " " + candidates["district_raw"].astype(str)
                           + " " + candidates["primary_domain"].astype(str) + " "
                           + candidates["top_drivers"].astype(str)).map(norm_name)
    candidates["_hits"] = candidates["_text"].map(lambda s: sum(t in s for t in terms))
    for _, candidate in candidates[candidates["_hits"] > 0].sort_values(
        ["_hits", "composite_percentile"], ascending=[False, False]
    ).head(top_k).iterrows():
        evidence.append({
            "source": "NFHS-5 district profile + KSHETRA-AI composite",
            "district": f"{candidate['district_raw']} / {candidate['state_raw']}",
            "fact": (f"percentile={candidate['composite_percentile']:.1f}; tier={candidate['risk_tier']}; "
                     f"main_domain={candidate['primary_domain']}; top_drivers={candidate['top_drivers']}"),
        })
    return evidence


def answer_query(question: str, profiles: pd.DataFrame, district: str | None = None,
                 state: str | None = None) -> dict[str, Any]:
    """Return a constrained answer plus evidence items; unsupported claims are not generated."""
    hindi = _is_hindi(question)
    row = _district_row(profiles, district, question, state)
    q = question.lower()
    evidence = retrieve_evidence(question, profiles, district, state=state)
    if row is None:
        if not evidence:
            msg = ("I couldn't retrieve a district record for that question. Select a district or ask about a named district in this dataset."
                   if not hindi else "इस प्रश्न के लिए जिला रिकॉर्ड नहीं मिला। जिला चुनें या इस डेटासेट के किसी जिले का नाम लिखें।")
        else:
            msg = ("I found related records, but not a single named district. Please specify the district for a grounded profile response."
                   if not hindi else "कुछ संबंधित रिकॉर्ड मिले, पर एक निश्चित जिला नहीं। कृपया जिला बताएं।")
        return {"answer": msg, "evidence": evidence, "district": None}

    name = f"{row['district_raw']}, {row['state_raw']}"
    pct = float(row["composite_percentile"])
    tier = str(row["risk_tier"])
    intent = "drivers" if any(w in q for w in ["why", "driver", "reason", "क्यों", "कारण"]) else (
        "scheme" if any(w in q for w in ["scheme", "coverage", "gap", "योजना", "कवरेज"]) else (
        "mpi" if any(w in q for w in ["mpi", "headcount", "census"]) else "summary"))

    if hindi:
        if intent == "drivers":
            answer = (f"{name} का KSHETRA-AI जोखिम स्तर {tier} है (जिला-वंचना percentile {pct:.0f}). "
                      f"मुख्य क्षेत्र: {row['primary_domain']}. सबसे बड़े संकेतक योगदान: {row['top_drivers']}. "
                      "ये संकेत survey indicators से बने हैं; स्थानीय सत्यापन आवश्यक है।")
        elif intent == "scheme":
            answer = (f"{name} के लिए उपलब्ध NFHS-5 सेवा-कवरेज proxies नीचे हैं। ये PMUY/SBM-G/PM-JAY की "
                      "लाभार्थी संख्या या पात्रता नहीं बताते। किसी योजना निर्णय से पहले प्रशासनिक रिकॉर्ड और स्थानीय सत्यापन देखें।")
        elif intent == "mpi":
            if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
                answer = (f"{name} के संदिग्ध स्रोत वाले Census-2011 MPI तालिका में HCR {row['mpi_hcr_pct']:.1f}% है "
                          f"(नाम मिलान: {row['mpi_match']}). इस मान की मूल स्रोत-पुष्टि आवश्यक है।")
            else:
                answer = f"{name} के लिए लिंक किया हुआ Census-2011 MPI रिकॉर्ड उपलब्ध नहीं है। मॉडल अनुमान को अवलोकित डेटा न मानें।"
        else:
            answer = (f"{name}: {tier} जोखिम स्तर, deprivation percentile {pct:.0f}, मुख्य क्षेत्र {row['primary_domain']}, "
                      f"विश्वास {row['confidence']}. यह स्क्रीनिंग संकेत है, गरीबी का प्रमाण नहीं। स्थानीय सत्यापन जरूरी है।")
    else:
        if intent == "drivers":
            answer = (f"{name} is in the {tier} screening tier (composite deprivation percentile {pct:.0f}). "
                      f"The largest domain is {row['primary_domain']}; top indicator drivers are {row['top_drivers']}. "
                      "These are survey-based signals, not proof of poverty; field verification is required.")
        elif intent == "scheme":
            answer = (f"Available NFHS-5 service-coverage proxies for {name} are listed below. They do not measure PMUY, SBM-G, "
                      "or PM-JAY enrollment or eligibility. Check administrative scheme records and verify locally before acting.")
        elif intent == "mpi":
            if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
                answer = (f"The supplied Census-2011 MPI candidate table reports an HCR of {row['mpi_hcr_pct']:.1f}% for {name} "
                          f"({row['mpi_match']} name match). The source provenance is undocumented and must be confirmed.")
            else:
                answer = f"No linked Census-2011 MPI record is available for {name}. Do not treat the model estimate as an observation."
        else:
            answer = (f"{name}: {tier} screening tier, composite deprivation percentile {pct:.0f}, main domain {row['primary_domain']}, "
                      f"data-completeness confidence {row['confidence']}. This is a risk-screening signal, not proof of poverty; "
                      "local field verification is required before action.")
    return {"answer": answer, "evidence": evidence, "district": name}
