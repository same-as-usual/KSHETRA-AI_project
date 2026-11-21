"""PRAGATI assistant: TF-IDF vector retrieval + constrained, evidence-grounded answer synthesis.

Implements a deterministic vector Retrieval-Augmented Generation (RAG) pipeline over the
707 district profiles and 6 programme-proxy definitions using scikit-learn TF-IDF n-gram
embeddings and cosine similarity, paired with grounded English/Hindi synthesis templates.
No external LLM API key is required; responses never invent unverified statistics.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemes import SCHEME_PROXIES, scheme_gap_frame
from .text import norm_name, norm_state

_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_ROMAN_HINDI_TOKENS = {
    "kyon", "kyo", "karan", "yojana", "kya", "jila", "garibi", "stithi", "vikas",
    "bijli", "pani", "swasthya", "shiksha", "poshan", "madad", "batao", "kaisa",
}


def _is_hindi(question: str) -> bool:
    if _DEVANAGARI.search(question):
        return True
    tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", question)}
    return bool(tokens & _ROMAN_HINDI_TOKENS)


def _district_row(profiles: pd.DataFrame, district: str | None, question: str, state: str | None = None) -> pd.Series | None:
    q = norm_name(question)
    # First check if the user explicitly named a district inside their question
    if q:
        matches = profiles[profiles["district_key"].map(lambda n: bool(n and len(n) >= 3 and n in q))]
        if len(matches) == 1:
            return matches.iloc[0]
        if len(matches) > 1:
            if state:
                state_matches = matches[matches["state_key"].eq(norm_state(state))]
                if len(state_matches) == 1:
                    return state_matches.iloc[0]
            return matches.loc[matches["district_key"].str.len().idxmax()]

    # Fall back to the UI-selected context district
    if district:
        wanted = norm_name(district)
        rows = profiles[profiles["district_key"].eq(wanted)]
        if state and len(rows) > 1:
            rows = rows[rows["state_key"].eq(norm_state(state))]
        if len(rows) >= 1:
            return rows.iloc[0]
    return None


def build_rag_corpus(profiles: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a structured knowledge corpus of district profiles and scheme-proxy cards for vector search."""
    docs: list[dict[str, Any]] = []
    for _, r in profiles.iterrows():
        d_name = f"{r.get('district_raw', '')} / {r.get('state_raw', '')}"
        mpi_str = (
            f"Census-2011 MPI HCR={float(r['mpi_hcr_pct']):.1f}% ({r.get('mpi_match')} match)"
            if pd.notna(r.get("mpi_hcr_pct", np.nan))
            else f"Proxy MPI estimate={float(r.get('proxy_mpi_hcr_pct', np.nan)):.1f}% (model output)"
        )
        text = (
            f"{r.get('district_raw', '')} {r.get('state_raw', '')} "
            f"tier {r.get('risk_tier', '')} domain {r.get('primary_domain', '')} "
            f"drivers {r.get('top_drivers', '')} percentile {r.get('composite_percentile', 0):.1f} "
            f"confidence {r.get('confidence', '')} {mpi_str}"
        )
        docs.append({
            "source": "NFHS-5 district profile + KSHETRA-AI composite",
            "district": d_name,
            "fact": (
                f"percentile={float(r.get('composite_percentile', np.nan)):.1f}; "
                f"tier={r.get('risk_tier')}; main_domain={r.get('primary_domain')}; "
                f"top_drivers={r.get('top_drivers')}; confidence={r.get('confidence')}; {mpi_str}"
            ),
            "_search_text": text,
        })
    for scheme in SCHEME_PROXIES:
        docs.append({
            "source": f"Programme proxy reference · {scheme.ministry}",
            "district": "All districts (NFHS-5 proxy definition)",
            "fact": f"{scheme.name} ({scheme.proxy_kind}): measured via '{scheme.proxy_measure_label}'. {scheme.interpretation}",
            "_search_text": f"{scheme.key} {scheme.name} {scheme.ministry} {scheme.proxy_kind} {scheme.proxy_measure_label} {scheme.gap_label} {scheme.interpretation}",
        })
    return docs


def vector_search_corpus(question: str, profiles: pd.DataFrame, top_k: int = 6) -> list[dict[str, Any]]:
    """Rank corpus chunks against the query using TF-IDF (word + char n-grams) and cosine similarity."""
    if profiles.empty or not question.strip():
        return []
    corpus = build_rag_corpus(profiles)
    texts = [doc["_search_text"] for doc in corpus]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(texts)
    q_vec = vec.transform([question])
    sims = cosine_similarity(q_vec, mat).ravel()
    top_indices = np.argsort(sims)[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in top_indices:
        score = float(sims[idx])
        if score <= 0.01:
            continue
        item = {k: v for k, v in corpus[idx].items() if not k.startswith("_")}
        item["similarity"] = round(score, 3)
        results.append(item)
    return results


def retrieve_evidence(question: str, profiles: pd.DataFrame, district: str | None = None,
                      top_k: int = 6, state: str | None = None) -> list[dict[str, Any]]:
    """Retrieve exact district facts + TF-IDF cosine-similarity scored evidence cards."""
    row = _district_row(profiles, district, question, state)
    evidence: list[dict[str, Any]] = []
    if row is not None:
        d_label = f"{row.get('district_raw','')} / {row.get('state_raw','')}"
        evidence.append({
            "source": "NFHS-5 district profile + KSHETRA-AI composite",
            "district": d_label,
            "fact": (f"percentile={row.get('composite_percentile', float('nan')):.1f}; "
                     f"tier={row.get('risk_tier')}; main_domain={row.get('primary_domain')}; "
                     f"top_drivers={row.get('top_drivers')}; confidence={row.get('confidence')}"),
            "similarity": 1.0,
        })
        if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
            evidence.append({
                "source": "Census 2011 MPI candidate table (provenance to confirm)",
                "district": d_label,
                "fact": f"MPI HCR={float(row['mpi_hcr_pct']):.1f}% ({row.get('mpi_match')} name match)",
                "similarity": 0.95,
            })
        gaps = scheme_gap_frame(pd.DataFrame([row])).dropna(subset=["gap_proxy_pct"])
        if not gaps.empty:
            # Rank district scheme proxies by relevance to question + gap magnitude
            gap_texts = [
                f"{g['scheme']} {g['proxy_kind']} {g['proxy_measure_label']} {g['gap_label']}"
                for _, g in gaps.iterrows()
            ]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            mat = vec.fit_transform(gap_texts)
            q_sim = cosine_similarity(vec.transform([question]), mat).ravel()
            gaps = gaps.copy()
            gaps["_sim"] = np.round(0.55 + 0.40 * q_sim + 0.05 * (gaps["gap_proxy_pct"].clip(0, 100) / 100.0), 3)
            gaps = gaps.sort_values(["_sim", "gap_proxy_pct"], ascending=[False, False])
            for _, gap in gaps.iterrows():
                evidence.append({
                    "source": f"NFHS-5 (2019-21) {gap['proxy_kind']}; {gap['scheme']}",
                    "district": d_label,
                    "fact": (f"{gap['proxy_measure_label']}={gap['proxy_measure_pct']:.1f}%; "
                             f"{gap['gap_label']}={gap['gap_proxy_pct']:.1f}%; not scheme enrollment"),
                    "similarity": float(gap["_sim"]),
                })
        return evidence[:top_k]

    # No single district matched: run TF-IDF vector search combined with token hits
    terms = {t for t in re.findall(r"[A-Z0-9]{3,}", norm_name(question))}
    if terms:
        candidates = profiles.copy()
        candidates["_text"] = (
            candidates["state_raw"].astype(str) + " " + candidates["district_raw"].astype(str)
            + " " + candidates["primary_domain"].astype(str) + " " + candidates["top_drivers"].astype(str)
        ).map(norm_name)
        candidates["_hits"] = candidates["_text"].map(lambda s: sum(t in s for t in terms))
        hit_rows = candidates[candidates["_hits"] > 0].sort_values(
            ["_hits", "composite_percentile"], ascending=[False, False]
        ).head(top_k)
        for _, candidate in hit_rows.iterrows():
            evidence.append({
                "source": "NFHS-5 district profile + KSHETRA-AI composite",
                "district": f"{candidate['district_raw']} / {candidate['state_raw']}",
                "fact": (f"percentile={candidate['composite_percentile']:.1f}; tier={candidate['risk_tier']}; "
                         f"main_domain={candidate['primary_domain']}; top_drivers={candidate['top_drivers']}"),
                "similarity": round(min(0.99, 0.45 + 0.15 * float(candidate["_hits"])), 3),
            })
        if evidence:
            return evidence[:top_k]

    return vector_search_corpus(question, profiles, top_k=top_k)


def answer_query(question: str, profiles: pd.DataFrame, district: str | None = None,
                 state: str | None = None) -> dict[str, Any]:
    """Return a constrained, evidence-grounded answer plus vector-retrieved evidence items."""
    hindi = _is_hindi(question)
    row = _district_row(profiles, district, question, state)
    q = question.lower()
    evidence = retrieve_evidence(question, profiles, district, state=state)
    if row is None:
        if not evidence:
            msg = ("I couldn't retrieve a district record for that question. Select a district or ask about a named district in this dataset."
                   if not hindi else "इस प्रश्न के लिए जिला रिकॉर्ड नहीं मिला। जिला चुनें या इस डेटासेट के किसी जिले का नाम लिखें।")
        else:
            msg = ("I found related records via TF-IDF similarity, but not a single named district. Please specify the district for a grounded profile response."
                   if not hindi else "कुछ संबंधित रिकॉर्ड मिले, पर एक निश्चित जिला नहीं। कृपया जिला बताएं।")
        return {"answer": msg, "evidence": evidence, "district": None, "intent": "unmatched"}

    name = f"{row['district_raw']}, {row['state_raw']}"
    pct = float(row["composite_percentile"])
    tier = str(row["risk_tier"])

    if any(w in q for w in ["why", "driver", "reason", "cause", "domain", "kyon", "kyo", "karan", "क्यों", "कारण", "मुख्य"]):
        intent = "drivers"
    elif any(w in q for w in ["scheme", "coverage", "gap", "pmuy", "sbm", "pmjay", "jal", "poshan", "samagra", "yojana", "योजना", "कवरेज", "लाभ"]):
        intent = "scheme"
    elif any(w in q for w in ["mpi", "headcount", "census", "cross-check", "xgboost", "model", "एमपीआई", "मॉडल"]):
        intent = "mpi"
    elif any(w in q for w in ["plan", "budget", "action", "interven", "allocate", "बजट", "उपाय", "योजना बजट"]):
        intent = "action"
    else:
        intent = "summary"

    gaps = scheme_gap_frame(pd.DataFrame([row])).dropna(subset=["gap_proxy_pct"])
    top_service_gaps = gaps[gaps["planner_eligible"]].sort_values("gap_proxy_pct", ascending=False).head(2)
    top_gap_summary = ", ".join(
        f"{g['scheme'].split('(')[0].strip()} ({g['gap_proxy_pct']:.1f}% uncovered proxy)"
        for _, g in top_service_gaps.iterrows()
    )

    if hindi:
        if intent == "drivers":
            answer = (f"{name} का KSHETRA-AI जोखिम स्तर {tier} है (जिला-वंचना percentile {pct:.0f}). "
                      f"मुख्य क्षेत्र: {row['primary_domain']}. सबसे बड़े संकेतक योगदान: {row['top_drivers']}. "
                      "ये संकेत NFHS-5 survey indicators से बने हैं; स्थानीय सत्यापन आवश्यक है।")
        elif intent == "scheme":
            answer = (f"{name} के लिए प्रमुख सेवा-अंतराल (NFHS-5 proxies): {top_gap_summary or 'नीचे तालिका देखें'}। "
                      "ये किसी योजना की लाभार्थी संख्या या पात्रता नहीं बताते। निर्णय से पहले प्रशासनिक MIS रिकॉर्ड और स्थानीय सत्यापन देखें।")
        elif intent == "mpi":
            if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
                answer = (f"{name} के संदिग्ध स्रोत वाले Census-2011 MPI तालिका में HCR {row['mpi_hcr_pct']:.1f}% है "
                          f"(नाम मिलान: {row['mpi_match']}). इस मान की मूल स्रोत-पुष्टि आवश्यक है।")
            else:
                answer = f"{name} के लिए लिंक किया हुआ Census-2011 MPI रिकॉर्ड उपलब्ध नहीं है। मॉडल अनुमान को अवलोकित डेटा न मानें।"
        elif intent == "action":
            answer = (f"{name} के लिए परिदृश्य योजना में सबसे बड़े सेवा-अंतराल {top_gap_summary} हैं। "
                      "Action Planner बजट को केवल चार तुलनीय सेवा-पहुँच proxies के अनुपात में बाँटता है; यह लागत या प्रभाव का पूर्वानुमान नहीं है।")
        else:
            answer = (f"{name}: {tier} जोखिम स्तर, deprivation percentile {pct:.0f}, मुख्य क्षेत्र {row['primary_domain']}, "
                      f"विश्वास {row['confidence']}. यह स्क्रीनिंग संकेत है, गरीबी का प्रमाण नहीं। स्थानीय सत्यापन जरूरी है।")
    else:
        if intent == "drivers":
            answer = (f"{name} is in the {tier} screening tier (composite deprivation percentile {pct:.0f}). "
                      f"The largest domain is {row['primary_domain']}; top indicator drivers are {row['top_drivers']}. "
                      "These are survey-based signals, not proof of poverty; field verification is required.")
        elif intent == "scheme":
            answer = (f"Available NFHS-5 service-access and outcome proxies for {name} are listed below "
                      f"(largest service-access gaps: {top_gap_summary or 'see evidence table'}). They do not measure scheme enrollment "
                      "or eligibility. Check current administrative records and verify locally before acting.")
        elif intent == "mpi":
            if pd.notna(row.get("mpi_hcr_pct", float("nan"))):
                answer = (f"The supplied Census-2011 MPI candidate table reports an HCR of {row['mpi_hcr_pct']:.1f}% for {name} "
                          f"({row['mpi_match']} name match). The source provenance is undocumented and must be confirmed.")
            else:
                answer = f"No linked Census-2011 MPI record is available for {name}. Do not treat the model estimate as an observation."
        elif intent == "action":
            answer = (f"For {name}, the largest NFHS-5 service-access gaps are {top_gap_summary}. "
                      "The Action Planner splits hypothetical budgets across the four service-access proxies in proportion to these gaps; "
                      "verify current MIS denominators and unit costs locally.")
        else:
            answer = (f"{name}: {tier} screening tier, composite deprivation percentile {pct:.0f}, main domain {row['primary_domain']}, "
                      f"data-completeness confidence {row['confidence']}. This is a risk-screening signal, not proof of poverty; "
                      "local field verification is required before action.")
    return {"answer": answer, "evidence": evidence, "district": name, "intent": intent}
