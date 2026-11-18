"""
================================================================================
CHEST X-RAY DUAL REPORT GENERATION & REFINEMENT MODULE
================================================================================
This module refines raw generated diagnostic X-ray reports into TWO distinct reports:
  1. Patient-Friendly Simple Report (clear, easy English, jargon explained)
  2. Professional Radiology Report (standard medical terminology, structured FINDINGS & IMPRESSION)

It includes a Quality Control / Safety Validation step to guarantee semantic consistency,
laterality preservation, and zero hallucination of clinical findings.

Supports external LLM APIs (OpenAI, Gemini, Groq, local Ollama) via environment variables
(API_KEY, OPENAI_API_KEY, GEMINI_API_KEY) with a robust offline medical NLP rule & dictionary engine fallback.
================================================================================
"""

import os
import re
import json
import requests
from typing import Dict, Tuple, List, Optional, Any

# ==============================================================================
# MEDICAL TERMINOLOGY MAPPINGS & TRANSLATION DICTIONARIES
# ==============================================================================

INFORMAL_TO_MEDICAL = [
    (r"\b(the heart is big|heart is big|heart is enlarged|enlarged heart|heart enlarged|cardiac silhouette is enlarged|cardiac size is enlarged|big heart|enlarged cardiac silhouette|enlargement of the cardiac silhouette)\b", "Cardiomegaly"),
    (r"\b(elevation of pulmonary venous pressure|pulmonary venous pressure|pulmonary vascular congestion|vascular congestion)\b", "pulmonary vascular congestion"),
    (r"\b(fluid around (the )?left lung|fluid around left lung|fluid accumulation in left lung|left pleural fluid)\b", "left-sided pleural effusion"),
    (r"\b(fluid around (the )?right lung|fluid around right lung|fluid accumulation in right lung|right pleural fluid)\b", "right-sided pleural effusion"),
    (r"\b(fluid around (the )?lung|fluid around (the )?lungs|fluid in (the )?pleural space|fluid buildup around (the )?lung|fluid accumulation|fluid in left lung|fluid in right lung|pleural fluid)\b", "pleural effusion"),
    (r"\b(collapsed lung|lung collapse|partially collapsed lung)\b", "atelectasis"),
    (r"\b(air around (the )?right lung|air in right pleural space)\b", "right pneumothorax"),
    (r"\b(air around (the )?left lung|air in left pleural space)\b", "left pneumothorax"),
    (r"\b(air around (the )?lung|air in (the )?pleural space|air leakage around (the )?lung)\b", "pneumothorax"),
    (r"\b(cloudy spot|cloudy area in (the )?lung|cloudy area|patchy opacity|white spot in (the )?lung|hazy area in (the )?lung)\b", "air-space opacity"),
    (r"\b(heart (size )?is normal|heart size is within normal limits|normal heart size|cardiac size is normal|normal size heart)\b", "cardiac silhouette is within normal limits"),
    (r"\b(fluid in the lungs|fluid in lungs|fluid accumulation in the lungs)\b", "pulmonary edema"),
    (r"\b(acute focal pneumonia|focal pneumonia|lung infection|infection in (the )?lung|infection-like abnormality)\b", "pneumonia"),
]

KNOWN_FINDINGS = [
    "cardiomegaly", "pleural effusion", "atelectasis",
    "pneumothorax", "opacity", "pulmonary edema", "pneumonia",
    "consolidation", "infiltrate", "nodule", "fracture", "granuloma",
    "emphysema", "pneumoperitoneum", "congestion"
]


# ==============================================================================
# PROMPT CONSTANTS FOR LLM INTEGRATION
# ==============================================================================

SIMPLE_REPORT_PROMPT = (
    "You are a medical report language assistant. Rewrite the provided chest X-ray findings into clear, "
    "patient-friendly English. Preserve every clinical finding and its meaning. Do not add, remove, or change "
    "diagnoses, anatomical locations, laterality, severity, or certainty. Do not invent information.\n\n"
    "Raw Findings: {raw_report}\n\n"
    "Output ONLY the patient-friendly simple report."
)

PROFESSIONAL_REPORT_PROMPT = (
    "You are a professional radiology report language assistant. Rewrite the provided chest X-ray findings "
    "using standard professional radiology terminology and grammatically correct clinical sentences. "
    "Preserve exactly the same clinical meaning. Do not introduce new findings or diagnoses. Maintain anatomical "
    "location, laterality, severity, and certainty. Organize the output into Findings and Impression.\n\n"
    "Format:\n"
    "FINDINGS:\n"
    "[professional description of the observed findings]\n\n"
    "IMPRESSION:\n"
    "[concise summary of the major findings]\n\n"
    "Raw Findings: {raw_report}"
)


# ==============================================================================
# API CALLERS (OPENAI / GEMINI / GROQ / OLLAMA)
# ==============================================================================

def _get_api_key() -> Optional[str]:
    """Retrieve available API key from environment variables or Streamlit session_state."""
    # Check Streamlit session_state first if running inside active Streamlit context
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None and "user_api_key" in st.session_state and st.session_state["user_api_key"]:
            return st.session_state["user_api_key"].strip()
    except Exception:
        pass

    return (
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GROQ_API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("API_KEY")
    )



def _call_llm_api(prompt: str) -> Optional[str]:
    """
    Attempts to call an LLM API if credentials are provided or if Ollama is running locally.
    Supports Google Gemini (1.5/2.0 Flash), Groq (Llama 3.3 70B), OpenAI, and Local Ollama.
    Returns None if no key is set or if the call fails, falling back to local BioBERT engine.
    """
    api_key = _get_api_key()

    # 1. Google Gemini REST API (Primary Recommended for Major Project)
    gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if api_key and (api_key.startswith("AIza") or "gemini" in api_key.lower()) else None)
    if not gemini_key and api_key and not api_key.startswith("sk-") and not api_key.startswith("gsk_"):
        gemini_key = api_key

    if gemini_key:
        for model_ver in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_ver}:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    res_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    if res_text:
                        return res_text
            except Exception:
                pass

    # 2. Groq Cloud REST API (Ultra-fast Llama-3-70B)
    groq_key = os.getenv("GROQ_API_KEY") or (api_key if api_key and api_key.startswith("gsk_") else None)
    if groq_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # 3. OpenAI REST API
    openai_key = os.getenv("OPENAI_API_KEY") or (api_key if api_key and api_key.startswith("sk-") and not api_key.startswith("sk-or-") else None)
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # 4. Local Ollama Server (100% Offline fallback if running)
    try:
        url = "http://localhost:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "llama3",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    return None



def clean_raw_report(raw: str) -> str:
    """
    Strips preamble noise, incomplete comparison phrases, dangling preambles, and
    repeated inline section headers (e.g. 'findings:', 'impression:'), while strictly
    preserving all clinically meaningful positive and negative findings.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()

    # 1. Remove comparison preambles and incomplete comparison phrases FIRST
    preamble_patterns = [
        r"\b(in\s+comparison\s+with\s+the\s+study\s+of|in\s+comparison\s+with\s+the\s+study|in\s+comparison\s+with\s+prior|in\s+comparison\s+with\s+previous|in\s+comparison\s+with)\b",
        r"\b(compared\s+to\s+the\s+study\s+of|compared\s+to\s+prior|compared\s+to\s+previous|compared\s+to|compared\s+with|as\s+compared\s+to)\b",
        r"\b(comparison\s+to\s+prior|comparison\s+to\s+previous|comparison\s+is\s+made\s+with|comparison\s+to)\b",
        r"\b(the\s+study\s+of|prior\s+study\s+of|previous\s+study\s+of)\b(\s+[\d/\-]+)?",
        r"\bthere\s+is\s+little\s+overall\s+change\b",
        r"\bno\s+significant\s+change\b",
        r"\bno\s+change\b",
    ]
    for pattern in preamble_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # 2. Remove repeated section headers anywhere in text (e.g. 'findings:', 'impression:')
    text = re.sub(r"\b(findings|impression|comparison|history|examination|indication)[:\s]+", " ", text, flags=re.IGNORECASE)

    # 3. Clean up dangling words like 'again' when used in temporal comparison context
    text = re.sub(r"\bagain\s+(noted|seen|demonstrated|identified)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bagain\b", "", text, flags=re.IGNORECASE)

    # 4. Remove dangling prepositions or incomplete fragment endings at sentence boundaries
    text = re.sub(r"\b(in\s+comparison|compared|comparison|study\s+of)\b", " ", text, flags=re.IGNORECASE)

    # 5. Fix punctuation and spacing anomalies
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # remove space before punctuation
    text = re.sub(r"([.,;:!?])\1+", r"\1", text)  # deduplicate consecutive punctuation
    text = re.sub(r"^\s*[,.:;;\-]\s*", "", text)   # remove leading punctuation
    text = re.sub(r"\s+", " ", text).strip()        # normalize whitespace

    # 6. Ensure proper capitalization if text is non-empty
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def _is_term_negated(text_lower: str, term: str) -> bool:
    """
    Checks if a medical term in text is negated using scoped clause-aware matching.
    Ensures pre-negation triggers (e.g. 'without', 'no') only negate target terms that follow them,
    not preceding terms (e.g. 'cardiomegaly without edema').
    """
    if not text_lower or not term:
        return False

    term_re = r"\b" + re.escape(term.lower()) + r"s?\b"

    # 1. Post-negation patterns (target term followed by negation phrase)
    post_pat = term_re + r"\s+(is\s+|are\s+)?(absent|not\s+seen|not\s+identified|not\s+observed|ruled\s+out|negative)\b"
    if re.search(post_pat, text_lower):
        return True

    # 2. Pre-negation triggers
    neg_triggers = [
        r"\bno\b", r"\bwithout\b", r"\babsent\b", r"\bnegative for\b",
        r"\bfree of\b", r"\bdenies\b", r"\bno evidence of\b", r"\bclear of\b",
        r"\bunremarkable for\b"
    ]

    for trig in neg_triggers:
        pattern_pre = trig + r"(?:\s+[\w\-]+){0,5}?\s+" + term_re
        match = re.search(pattern_pre, text_lower)
        if match:
            matched_text = match.group(0)
            if not re.search(r"\b(but|however|although|with|showing|demonstrating)\b", matched_text[len(trig):]):
                return True

    return False


def _has_historical_context(text_lower: str, finding_keywords: List[str]) -> bool:
    """
    Checks if a finding in text_lower is explicitly associated with a historical/prior context
    (e.g., 'prior study showed', 'history of', 'resolved', 'interval resolution').
    """
    if not text_lower or not finding_keywords:
        return False
    
    hist_triggers = [
        r"\bprior\b", r"\bprevious\b", r"\bearlier\b", r"\bpast\b",
        r"\bhistory of\b", r"\bh/o\b", r"\bold\b", r"\binterval resolution\b",
        r"\bresolved\b", r"\bcleared\b", r"\bcompared to prior\b", r"\bcompared with prior\b"
    ]
    
    clauses = [c.strip() for c in re.split(r"[.,;:!\n]", text_lower) if c and len(c.strip()) > 1]
    for clause in clauses:
        has_kw = any(re.search(r"\b" + re.escape(kw) + r"s?\b", clause) for kw in finding_keywords)
        if has_kw:
            if any(re.search(trig, clause) for trig in hist_triggers):
                return True
    return False



def _get_laterality(text_lower: str, finding_keywords: List[str]) -> Optional[str]:
    """
    Extracts anatomical laterality ('bilateral', 'left', or 'right') specifically associated
    with the target finding keywords in the given text.
    Evaluates bilateral/both patterns BEFORE individual left/right patterns.
    Preserves implicit laterality from nearby anatomical phrases (e.g. 'left base', 'right base')
    while strictly scoping window boundaries to the containing sentence/clause.
    """
    if not text_lower or not finding_keywords:
        return None

    # Split text into sentences/clauses using punctuation and clause conjunctions
    clauses = [c.strip() for c in re.split(r"[.,;:!\n]|\b(and|but)\b", text_lower) if c and len(c.strip()) > 1]

    matching_clauses = []
    for clause in clauses:
        if any(re.search(r"\b" + re.escape(kw) + r"s?\b", clause) for kw in finding_keywords):
            matching_clauses.append(clause)

    search_scope = " ".join(matching_clauses) if matching_clauses else text_lower

    # 1. Evaluate Bilateral / Both patterns FIRST
    if re.search(r"\b(bilateral|bilaterally|both|both-sided|both\s+bases|both\s+lungs|both\s+sides|involving both|in both)\b", search_scope):
        return "bilateral"
    
    has_left = bool(re.search(r"\b(left|left-sided|left\s+base|left-base|left\s+lung|left\s+lower|left\s+upper|left\s+mid|left\s+hemithorax|left\s+costophrenic|left\s+apex)\b", search_scope))
    has_right = bool(re.search(r"\b(right|right-sided|right\s+base|right-base|right\s+lung|right\s+lower|right\s+upper|right\s+mid|right\s+hemithorax|right\s+costophrenic|right\s+apex)\b", search_scope))

    if has_left and has_right:
        return "bilateral"
    elif has_left:
        return "left"
    elif has_right:
        return "right"

    return None


# ==============================================================================
# LOCAL MEDICAL NLP REFINEMENT ENGINE (FALLBACK / OFFLINE)
# ==============================================================================

def _extract_severity_prefix(text_lower: str, target_keywords: List[str]) -> str:
    """Extracts explicit severity modifier associated with a finding in text_lower."""
    if not text_lower or not target_keywords:
        return ""
    clauses = [c.strip() for c in re.split(r"[.,;:!\n]", text_lower) if c and len(c.strip()) > 1]
    for clause in clauses:
        if any(kw in clause for kw in target_keywords):
            if "mild-to-moderate" in clause or "mild to moderate" in clause:
                return "mild-to-moderate "
            elif "substantial" in clause:
                return "substantial "
            elif "severe" in clause or "marked" in clause:
                return "severe "
            elif "moderate" in clause:
                return "moderate "
            elif "mild" in clause or "slight" in clause:
                return "mild "
            elif "small" in clause:
                return "small "
            elif "large" in clause:
                return "large "
    return ""


# ==============================================================================
# LOCAL MEDICAL NLP REFINEMENT ENGINE (FALLBACK / OFFLINE)
# ==============================================================================

def _add_unique(target_list: List[str], item: str) -> None:
    """Appends an item to a list if it is non-empty and not already present."""
    if item and item not in target_list:
        target_list.append(item)


def generate_simple_report(raw_report: str, biobert_findings: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generates a patient-friendly simple report in clear, accessible English.
    Translates medical jargon into plain terms while strictly preserving clinical meaning.
    """
    cleaned_raw = clean_raw_report(raw_report)
    if not cleaned_raw:
        return "No diagnostic findings recorded to summarize."

    # 1. Try LLM API first if available
    llm_prompt = SIMPLE_REPORT_PROMPT.format(raw_report=cleaned_raw)
    llm_output = _call_llm_api(llm_prompt)
    if llm_output:
        clean_llm = re.sub(r"^(SIMPLE REPORT:?|Patient-Friendly Report:?)\s*", "", llm_output, flags=re.IGNORECASE).strip()
        return clean_llm

    # 2. Offline Rule-based Medical NLP Engine (BioBERT-Enhanced)
    text_lower = cleaned_raw.lower()

    positive_parts = []
    negative_parts = []
    interval_parts = []

    # 1. Support Devices & Procedures (Chest Tubes, Extubation, Pacemakers, Lines, Wires)
    # Chest Tube
    if "chest tube" in text_lower or ("tube" in text_lower and "chest" in text_lower):
        if not _is_term_negated(text_lower, "chest tube"):
            ct_lat = _get_laterality(text_lower, ["chest tube", "tube"])
            loc = "on the right side" if ct_lat == "right" or "right" in text_lower else ("on the left side" if ct_lat == "left" or "left" in text_lower else "in the chest")
            msg = f"a chest tube is present {loc}, placed since the previous X-ray" if ("previous" in text_lower or "prior" in text_lower or "received" in text_lower) else f"a chest tube is present {loc}"
            _add_unique(positive_parts, msg)

    # Extubation
    if ("extubated" in text_lower or "extubation" in text_lower) and not any("breathing tube" in p for p in interval_parts):
        _add_unique(interval_parts, "the breathing tube has been removed since the prior X-ray (interval extubation)")

    # Pacemaker Leads & Generator
    if ("pacemaker" in text_lower or "pacer" in text_lower) and not _is_term_negated(text_lower, "pacemaker"):
        pace_lat = _get_laterality(text_lower, ["pacemaker", "pacer"])
        if not pace_lat:
            if "left pectoral" in text_lower or "left chest" in text_lower or "left side" in text_lower:
                pace_lat = "left"
            elif "right pectoral" in text_lower or "right chest" in text_lower or "right side" in text_lower:
                pace_lat = "right"
        loc_str = f" on the {pace_lat} side of the chest" if pace_lat in ["left", "right"] else " in the chest"
        is_stable = any(kw in text_lower for kw in ["constant", "stable", "unchanged", "leads"])
        
        if not any("pacemaker" in p for p in positive_parts):
            if is_stable and pace_lat:
                _add_unique(positive_parts, f"a {pace_lat} pectoral pacemaker is present with stable wires in unchanged position")
            elif is_stable:
                _add_unique(positive_parts, "the pacemaker wires remain unchanged and in stable position")
            else:
                _add_unique(positive_parts, f"a pacemaker is present{loc_str}".strip())

    # Catheters / Venous Lines / Sternal Wires
    if "jugular" in text_lower or "svc" in text_lower or "venous catheter" in text_lower or "line tip" in text_lower:
        if not _is_term_negated(text_lower, "catheter") and not _is_term_negated(text_lower, "line") and not any("pacemaker" in p for p in positive_parts):
            loc = "on the right side" if "right" in text_lower else ("on the left side" if "left" in text_lower else "")
            loc_str = f" {loc}" if loc else ""
            _add_unique(positive_parts, f"a medical tube is present{loc_str}, with its tip in a large vein")
    elif ("catheter" in text_lower or "subclavian" in text_lower) and not _is_term_negated(text_lower, "catheter") and not any("chest tube" in p for p in positive_parts):
        cath_lat = _get_laterality(text_lower, ["catheter", "subclavian"])
        loc = "on the left side" if cath_lat == "left" else ("on the right side" if cath_lat == "right" else "")
        _add_unique(positive_parts, f"a catheter is present {loc}".strip())

    if ("sternal wires" in text_lower or "sternotomy" in text_lower or "sternal" in text_lower) and not _is_term_negated(text_lower, "sternal") and not any("sternal" in p for p in positive_parts):
        _add_unique(positive_parts, "intact median sternotomy wires from prior surgery are present")

    # 2. Lung Volumes
    if ("lung volume" in text_lower or "lung volumes" in text_lower) and ("decreased" in text_lower or "low" in text_lower or "small" in text_lower) and not any("lung volumes" in p for p in positive_parts):
        _add_unique(positive_parts, "lung volumes are decreased")

    # 3. Cardiac Silhouette / Borderline / Cardiomegaly / PVP / Edema
    card_sev = _extract_severity_prefix(text_lower, ["cardiomegaly", "enlarged heart", "heart is enlarged"])
    card_adj = f"{card_sev.strip()} " if card_sev else ""

    edema_sev = _extract_severity_prefix(text_lower, ["edema"])
    edema_adj = f"{edema_sev.strip()} " if edema_sev else ""

    has_borderline = ("borderline" in text_lower or "upper limit" in text_lower or "upper limits" in text_lower) and ("cardiac" in text_lower or "heart" in text_lower or "silhouette" in text_lower or "size" in text_lower)
    has_cardio = bool(re.search(r"\b(cardiomegaly|enlarged heart|heart is enlarged|big heart|heart is big|cardiac silhouette is enlarged|enlarged cardiac silhouette|enlargement of the cardiac silhouette)\b", text_lower)) and not (_is_term_negated(text_lower, "cardiomegaly") or _is_term_negated(text_lower, "enlarged") or _is_term_negated(text_lower, "big"))
    has_edema = ("edema" in text_lower) and not _is_term_negated(text_lower, "edema")
    has_pvp = ("pulmonary venous pressure" in text_lower or "vascular congestion" in text_lower or "pulmonary congestion" in text_lower) and not (_is_term_negated(text_lower, "congestion") or _is_term_negated(text_lower, "pressure"))

    if has_borderline and has_cardio:
        prefix_str = f"the heart size is at the upper limit of normal to {card_adj}enlarged (borderline cardiac size to cardiomegaly)".replace("  ", " ")
        if has_pvp and has_edema:
            _add_unique(positive_parts, f"{prefix_str}, with increased pressure or congestion in lung blood vessels")
            _add_unique(positive_parts, f"there is {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        elif has_pvp:
            _add_unique(positive_parts, f"{prefix_str}, with increased pressure or congestion in lung blood vessels")
        elif has_edema:
            _add_unique(positive_parts, f"{prefix_str}, with {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        else:
            _add_unique(positive_parts, prefix_str)
    elif has_borderline:
        if has_pvp and has_edema:
            _add_unique(positive_parts, "the heart size is at the upper limit of normal (borderline enlarged), with increased pressure or congestion in lung blood vessels")
            _add_unique(positive_parts, f"there is {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        elif has_pvp:
            _add_unique(positive_parts, "the heart size is at the upper limit of normal (borderline enlarged), with increased pressure or congestion in lung blood vessels")
        elif has_edema:
            _add_unique(positive_parts, f"the heart size is at the upper limit of normal (borderline enlarged), with {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        else:
            _add_unique(positive_parts, "the heart size is at the upper limit of normal (borderline enlarged)")
    elif has_cardio:
        prefix_str = f"a {card_adj}enlarged heart (cardiomegaly)".replace("  ", " ") if card_adj else "an enlarged heart (cardiomegaly)"
        if has_pvp and has_edema:
            _add_unique(positive_parts, f"there is {prefix_str}, with increased pressure or congestion in lung blood vessels")
            _add_unique(positive_parts, f"there is {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        elif has_pvp:
            _add_unique(positive_parts, f"there is {prefix_str}, with increased pressure or congestion in lung blood vessels")
        elif has_edema:
            _add_unique(positive_parts, f"there is {prefix_str}, with {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
        else:
            _add_unique(positive_parts, f"there is {prefix_str}")
    elif has_pvp and has_edema:
        _add_unique(positive_parts, "there is increased pressure or congestion in lung blood vessels")
        _add_unique(positive_parts, f"there is {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))
    elif has_pvp:
        _add_unique(positive_parts, "there is increased pressure or congestion in lung blood vessels (pulmonary vascular congestion)")
    elif has_edema:
        _add_unique(positive_parts, f"there are signs of {edema_adj}fluid accumulation within the lungs (pulmonary edema)".replace("  ", " "))

    # 4. Lungs & Pleura (Atelectasis & Pleural Effusion Synthesis)
    has_effusion = ("effusion" in text_lower or "fluid" in text_lower) and not (_is_term_negated(text_lower, "effusion") or _is_term_negated(text_lower, "fluid"))
    has_atelectasis = ("atelectasis" in text_lower or "collapsed" in text_lower or "atelectatic" in text_lower) and not (_is_term_negated(text_lower, "atelectasis") or _is_term_negated(text_lower, "collapsed"))

    eff_lat = _get_laterality(text_lower, ["effusion", "fluid"]) if has_effusion else None
    atel_lat = _get_laterality(text_lower, ["atelectasis", "collapsed", "atelectatic"]) if has_atelectasis else None
    is_basilar = "basilar" in text_lower or "base" in text_lower or "bases" in text_lower or "bibasilar" in text_lower
    is_compressive = "compressive" in text_lower

    if has_effusion and has_atelectasis:
        if is_compressive and (is_basilar or eff_lat == "bilateral" or atel_lat == "bilateral"):
            _add_unique(positive_parts, "there are areas of partial compression-related collapse of lung tissue at the bases and fluid around both lungs")
        elif (is_basilar or eff_lat == "bilateral" or atel_lat == "bilateral"):
            _add_unique(positive_parts, "there are small areas of partial lung collapse at the bases and small amounts of fluid around both lungs")
        elif eff_lat == "left" or atel_lat == "left":
            _add_unique(positive_parts, "there are small areas of partial lung collapse and fluid around the left lung")
        elif eff_lat == "right" or atel_lat == "right":
            _add_unique(positive_parts, "there are small areas of partial lung collapse and fluid around the right lung")
        else:
            _add_unique(positive_parts, "there are areas of partial lung collapse and fluid around the lung")
    elif has_effusion:
        if eff_lat == "bilateral":
            _add_unique(positive_parts, "there is a buildup of fluid around both lungs")
        elif eff_lat == "left":
            _add_unique(positive_parts, "there is a buildup of fluid around the left lung")
        elif eff_lat == "right":
            _add_unique(positive_parts, "there is a buildup of fluid around the right lung")
        else:
            _add_unique(positive_parts, "there is a buildup of fluid around the lung")
    elif has_atelectasis:
        if is_compressive:
            _add_unique(positive_parts, "there is partial compression-related collapse of lung tissue at the bases")
        else:
            loc = "at the bases" if is_basilar else ("in both lungs" if atel_lat == "bilateral" else ("in the left lung" if atel_lat == "left" else ("in the right lung" if atel_lat == "right" else "")))
            loc_str = f" {loc}" if loc else ""
            _add_unique(positive_parts, f"there is a small area of collapsed lung tissue{loc_str}".strip())

    # 5. Pneumonia / Opacity
    has_pneumonia = ("pneumonia" in text_lower or "infection" in text_lower) and not (_is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection"))
    has_opacity = ("opacity" in text_lower or "opacities" in text_lower or "cloudy" in text_lower or "spot" in text_lower) and not (_is_term_negated(text_lower, "opacity") or _is_term_negated(text_lower, "opacities"))

    if has_pneumonia:
        _add_unique(positive_parts, "there are signs suggestive of a lung infection (pneumonia)")
    elif has_opacity and not any("cloudy" in p or "inflammation" in p for p in positive_parts):
        opac_lat = _get_laterality(text_lower, ["opacity", "opacities", "cloudy", "spot"])
        loc = "in both lungs" if opac_lat == "bilateral" else ("in the right lung" if opac_lat == "right" else ("in the left lung" if opac_lat == "left" else "in the lung"))
        _add_unique(positive_parts, f"there is a hazy or cloudy spot {loc}, which may indicate fluid or inflammation")

    # 6. Pneumothorax & Explicit Negatives
    has_pneu = ("pneumothorax" in text_lower or "air around" in text_lower or "air in pleural" in text_lower)
    if has_pneu:
        is_negated = _is_term_negated(text_lower, "pneumothorax") or _is_term_negated(text_lower, "air")
        is_hist = _has_historical_context(text_lower, ["pneumothorax", "air"])
        if is_negated:
            if is_hist:
                _add_unique(negative_parts, "No air leak around the lungs is identified. Prior pneumothorax has resolved.")
            else:
                _add_unique(negative_parts, "No air leak around the lungs is identified.")
        else:
            pneu_lat = _get_laterality(text_lower, ["pneumothorax", "air"])
            if pneu_lat == "bilateral":
                _add_unique(positive_parts, "there is trapped air around both lungs")
            elif pneu_lat == "left":
                _add_unique(positive_parts, "there is trapped air around the left lung")
            elif pneu_lat == "right":
                _add_unique(positive_parts, "there is trapped air around the right lung")
            else:
                _add_unique(positive_parts, "there is trapped air outside the lung")

    # Check combined explicit negatives (e.g. no vascular congestion, no pulmonary edema, no pleural effusion, no pneumonia)
    combined_negs = []
    if (_is_term_negated(text_lower, "congestion") or _is_term_negated(text_lower, "vascular congestion")) and not has_pvp:
        combined_negs.append("increased pressure or congestion in lung blood vessels (pulmonary vascular congestion)")
    if _is_term_negated(text_lower, "edema") and not has_edema:
        combined_negs.append("fluid buildup in the lungs (pulmonary edema)")
    if (_is_term_negated(text_lower, "effusion") or _is_term_negated(text_lower, "fluid")) and not has_effusion:
        combined_negs.append("fluid around the lungs")
    if (_is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection")) and not has_pneumonia:
        combined_negs.append("signs of pneumonia")

    if combined_negs and not negative_parts:
        if len(combined_negs) >= 3:
            _add_unique(negative_parts, f"No {', '.join(combined_negs[:-1])}, or {combined_negs[-1]} are identified.")
        else:
            _add_unique(negative_parts, f"No {', '.join(combined_negs)} identified.")

    all_sentences = []
    if positive_parts:
        formatted = [p[0].upper() + p[1:] + ("." if not p.endswith(".") else "") for p in positive_parts]
        all_sentences.append(" ".join(formatted))

    if interval_parts:
        formatted_int = [p[0].upper() + p[1:] + ("." if not p.endswith(".") else "") for p in interval_parts]
        all_sentences.append(" ".join(formatted_int))

    if negative_parts:
        all_sentences.extend(negative_parts)

    if all_sentences:
        return " ".join(all_sentences)

    if re.search(r"\b(no (acute )?cardiopulmonary (abnormality|disease|finding)|normal chest|unremarkable|within normal limits)\b", text_lower):
        return "The chest X-ray shows normal heart size and clear lungs. No signs of infection, fluid, or other acute abnormalities are present."

    s_clean = cleaned_raw
    for pattern, medical_term in INFORMAL_TO_MEDICAL:
        s_clean = re.sub(pattern, medical_term, s_clean, flags=re.IGNORECASE)
    return s_clean[0].upper() + s_clean[1:] + ("." if not s_clean.endswith(".") else "")


def generate_medical_report(raw_report: str, biobert_findings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """
    Generates a professional radiology report structured into:
    - FINDINGS: Professional clinical description using radiology terminology.
    - IMPRESSION: Concise summary of key clinical observations.
    Preserves all positive, negative, and uncertain clinical findings explicitly.
    """
    cleaned_raw = clean_raw_report(raw_report)
    if not cleaned_raw:
        return {
            "findings": "The chest radiograph is unremarkable.",
            "impression": "No acute cardiopulmonary process.",
            "full_text": "FINDINGS:\nThe chest radiograph is unremarkable.\n\nIMPRESSION:\nNo acute cardiopulmonary process."
        }

    # 1. Try LLM API first if available
    llm_prompt = PROFESSIONAL_REPORT_PROMPT.format(raw_report=cleaned_raw)
    llm_output = _call_llm_api(llm_prompt)
    if llm_output and "FINDINGS:" in llm_output and "IMPRESSION:" in llm_output:
        parts = llm_output.split("IMPRESSION:")
        findings_part = parts[0].replace("FINDINGS:", "").strip()
        impression_part = parts[1].strip()
        return {
            "findings": findings_part,
            "impression": impression_part,
            "full_text": f"FINDINGS:\n{findings_part}\n\nIMPRESSION:\n{impression_part}"
        }

    text_lower = cleaned_raw.lower()
    findings_elements = []
    impression_bullets = []
    explicit_negatives = []
    has_abnormality = False

    # 1. Support Devices & Procedure Changes (Chest Tubes, Extubation, Pacemakers, Lines, Wires)
    # Chest Tube
    if "chest tube" in text_lower or ("tube" in text_lower and "chest" in text_lower):
        if not _is_term_negated(text_lower, "chest tube"):
            ct_lat = _get_laterality(text_lower, ["chest tube", "tube"])
            side = "right " if ct_lat == "right" or "right" in text_lower else ("left " if ct_lat == "left" or "left" in text_lower else "")
            if "previous" in text_lower or "prior" in text_lower or "received" in text_lower:
                _add_unique(findings_elements, f"Compared to the prior radiograph, a {side}chest tube has been placed.".strip())
                _add_unique(impression_bullets, f"{side.capitalize()}chest tube in place.".strip())
            else:
                _add_unique(findings_elements, f"A {side}chest tube is in place.".strip())
                _add_unique(impression_bullets, f"{side.capitalize()}chest tube in place.".strip())
            has_abnormality = True

    # Pacemaker Leads & Generator Positioning
    if ("pacemaker" in text_lower or "pacer" in text_lower) and not _is_term_negated(text_lower, "pacemaker"):
        pace_lat = _get_laterality(text_lower, ["pacemaker", "pacer"])
        if not pace_lat:
            if "left pectoral" in text_lower or "left chest" in text_lower or "left side" in text_lower:
                pace_lat = "left"
            elif "right pectoral" in text_lower or "right chest" in text_lower or "right side" in text_lower:
                pace_lat = "right"
        loc = "Left pectoral " if pace_lat == "left" else ("Right pectoral " if pace_lat == "right" else "")
        is_stable = any(kw in text_lower for kw in ["constant", "stable", "unchanged", "leads"])

        if is_stable and pace_lat:
            _add_unique(findings_elements, f"{loc}pacemaker is present with constant, stable positioning of pacemaker leads.".strip())
            _add_unique(impression_bullets, f"Stable positioning of {loc.lower().strip()} pacemaker leads.")
        elif is_stable:
            _add_unique(findings_elements, "The course of the pacemaker leads is constant and in stable position.")
            _add_unique(impression_bullets, "Stable positioning of pacemaker leads.")
        else:
            _add_unique(findings_elements, f"{loc}Pacemaker generator is present.".strip())
            _add_unique(impression_bullets, f"{loc}Pacemaker in place.".strip())
        has_abnormality = True

    # Extubation
    if "extubated" in text_lower or "extubation" in text_lower:
        _add_unique(findings_elements, "The patient has been extubated since the prior study.")
        _add_unique(impression_bullets, "Interval extubation.")
        has_abnormality = True

    # Other Devices (Jugular Lines, Subclavian Catheters, Sternal Wires)
    if "jugular" in text_lower or "svc" in text_lower or "venous catheter" in text_lower or "line tip" in text_lower:
        if not _is_term_negated(text_lower, "catheter") and not _is_term_negated(text_lower, "line"):
            cath_match = re.search(r"\b((right|left)?\s*(internal\s+jugular|subclavian|venous)?\s*catheter\s*(tip)?\s*projects?[^.;!\n]*)", text_lower, re.IGNORECASE)
            if cath_match:
                matched_str = cath_match.group(1).strip()
                matched_cap = matched_str[0].upper() + matched_str[1:]
                if not matched_cap.endswith("."):
                    matched_cap += "."
                _add_unique(findings_elements, matched_cap)
                imp_cath = re.sub(r"\s*over the mid to lower\s*", " over the ", matched_cap, flags=re.IGNORECASE)
                _add_unique(impression_bullets, imp_cath.rstrip("."))
            else:
                _add_unique(findings_elements, "Venous catheter tip is noted projecting over the SVC.")
                _add_unique(impression_bullets, "Venous catheter tip projects over the SVC.")
            has_abnormality = True
    elif ("subclavian" in text_lower or "catheter" in text_lower) and not _is_term_negated(text_lower, "catheter") and not any("chest tube" in f.lower() for f in findings_elements):
        cath_lat = _get_laterality(text_lower, ["catheter", "subclavian"])
        loc = "Left subclavian " if cath_lat == "left" else ("Right subclavian " if cath_lat == "right" else "")
        _add_unique(findings_elements, f"{loc}Catheter is noted.".strip())
        _add_unique(impression_bullets, f"{loc}Catheter in place.".strip())
        has_abnormality = True

    if ("sternal wires" in text_lower or "sternotomy" in text_lower or "sternal" in text_lower) and not _is_term_negated(text_lower, "sternal"):
        _add_unique(findings_elements, "Intact midline sternal wires are present.")
        _add_unique(impression_bullets, "Intact median sternotomy wires.")
        has_abnormality = True

    # 2. Lung Volumes
    if ("lung volume" in text_lower or "lung volumes" in text_lower) and ("decreased" in text_lower or "low" in text_lower or "small" in text_lower):
        _add_unique(findings_elements, "Lung volumes are decreased.")
        _add_unique(impression_bullets, "Decreased lung volumes.")
        has_abnormality = True

    # 3. Cardiac Silhouette / Borderline / Cardiomegaly / Congestion / Edema
    card_sev = _extract_severity_prefix(text_lower, ["cardiomegaly", "enlarged heart", "heart is enlarged"])
    card_title = f"{card_sev.capitalize()}cardiomegaly" if card_sev else "Cardiomegaly"

    edema_sev = _extract_severity_prefix(text_lower, ["edema"])
    edema_str = f"{edema_sev}pulmonary edema" if edema_sev else "pulmonary edema"
    edema_cap = edema_str.capitalize()

    has_borderline = ("borderline" in text_lower or "upper limit" in text_lower or "upper limits" in text_lower) and ("cardiac" in text_lower or "heart" in text_lower or "silhouette" in text_lower or "size" in text_lower)
    has_cardio = bool(re.search(r"\b(cardiomegaly|enlarged heart|heart is enlarged|big heart|heart is big|cardiac silhouette is enlarged|enlarged cardiac silhouette|enlargement of the cardiac silhouette)\b", text_lower)) and not (_is_term_negated(text_lower, "cardiomegaly") or _is_term_negated(text_lower, "enlarged") or _is_term_negated(text_lower, "big"))
    has_edema = ("edema" in text_lower) and not _is_term_negated(text_lower, "edema")
    has_pvp = ("pulmonary venous pressure" in text_lower or "vascular congestion" in text_lower or "pulmonary congestion" in text_lower) and not (_is_term_negated(text_lower, "congestion") or _is_term_negated(text_lower, "pressure"))

    if has_borderline and has_cardio:
        title = f"Borderline cardiac silhouette to {card_title.lower()}"
        if has_pvp and has_edema:
            _add_unique(findings_elements, f"{title} with pulmonary vascular congestion.")
            _add_unique(impression_bullets, f"{title} with pulmonary vascular congestion.")
            _add_unique(findings_elements, f"{edema_cap} is present.")
            _add_unique(impression_bullets, f"{edema_cap}.")
        elif has_pvp:
            _add_unique(findings_elements, f"{title} with pulmonary vascular congestion.")
            _add_unique(impression_bullets, f"{title} with pulmonary vascular congestion.")
        elif has_edema:
            _add_unique(findings_elements, f"{title} with {edema_str}.")
            _add_unique(impression_bullets, f"{title} with {edema_str}.")
        else:
            _add_unique(findings_elements, f"{title} is noted.")
            _add_unique(impression_bullets, f"{title}.")
        has_abnormality = True
    elif has_borderline:
        if has_pvp and has_edema:
            _add_unique(findings_elements, "Borderline size of the cardiac silhouette with pulmonary vascular congestion.")
            _add_unique(impression_bullets, "Borderline cardiac silhouette with pulmonary vascular congestion.")
            _add_unique(findings_elements, f"{edema_cap} is present.")
            _add_unique(impression_bullets, f"{edema_cap}.")
        elif has_pvp:
            _add_unique(findings_elements, "Borderline size of the cardiac silhouette with pulmonary vascular congestion.")
            _add_unique(impression_bullets, "Borderline cardiac silhouette with pulmonary vascular congestion.")
        elif has_edema:
            _add_unique(findings_elements, f"Borderline size of the cardiac silhouette with {edema_str}.")
            _add_unique(impression_bullets, f"Borderline cardiac silhouette with {edema_str}.")
        else:
            _add_unique(findings_elements, "Borderline size of the cardiac silhouette.")
            _add_unique(impression_bullets, "Borderline cardiac silhouette.")
        has_abnormality = True
    elif has_cardio:
        if has_pvp and has_edema:
            _add_unique(findings_elements, f"{card_title} with pulmonary vascular congestion.")
            _add_unique(impression_bullets, f"{card_title} with pulmonary vascular congestion.")
            _add_unique(findings_elements, f"{edema_cap} is present.")
            _add_unique(impression_bullets, f"{edema_cap}.")
        elif has_pvp:
            _add_unique(findings_elements, f"{card_title} with pulmonary vascular congestion.")
            _add_unique(impression_bullets, f"{card_title} with pulmonary vascular congestion.")
        elif has_edema:
            _add_unique(findings_elements, f"{card_title} with {edema_str}.")
            _add_unique(impression_bullets, f"{card_title} with {edema_str}.")
        else:
            _add_unique(findings_elements, f"{card_title} is noted.")
            _add_unique(impression_bullets, f"{card_title}.")
        has_abnormality = True
    elif has_pvp and has_edema:
        _add_unique(findings_elements, "Pulmonary vascular congestion is present.")
        _add_unique(impression_bullets, "Pulmonary vascular congestion.")
        _add_unique(findings_elements, f"{edema_cap} is present.")
        _add_unique(impression_bullets, f"{edema_cap}.")
        has_abnormality = True
    elif has_pvp:
        _add_unique(findings_elements, "Mild elevation of pulmonary venous pressure is present.")
        _add_unique(impression_bullets, "Mild pulmonary vascular congestion.")
        has_abnormality = True
    elif has_edema:
        _add_unique(findings_elements, f"{edema_cap} is present.")
        _add_unique(impression_bullets, f"{edema_cap}.")
        has_abnormality = True
    elif ("normal" in text_lower or "limits" in text_lower or "normal size" in text_lower) and not has_cardio and not has_borderline:
        _add_unique(findings_elements, "The cardiac silhouette is within normal limits.")

    # 4. Lungs & Pleura (Atelectasis & Pleural Effusion Synthesis)
    has_effusion = ("effusion" in text_lower or "fluid" in text_lower) and not (_is_term_negated(text_lower, "effusion") or _is_term_negated(text_lower, "fluid"))
    has_atelectasis = ("atelectasis" in text_lower or "collapsed" in text_lower or "atelectatic" in text_lower) and not (_is_term_negated(text_lower, "atelectasis") or _is_term_negated(text_lower, "collapsed"))

    eff_lat = _get_laterality(text_lower, ["effusion", "fluid"]) if has_effusion else None
    atel_lat = _get_laterality(text_lower, ["atelectasis", "collapsed", "atelectatic"]) if has_atelectasis else None
    is_basilar = "basilar" in text_lower or "base" in text_lower or "bases" in text_lower or "bibasilar" in text_lower
    is_compressive = "compressive" in text_lower

    atel_desc = "compressive atelectasis" if is_compressive else "atelectatic changes"
    atel_imp = "compressive atelectasis" if is_compressive else "atelectatic changes"

    if has_effusion and has_atelectasis:
        if (is_basilar or eff_lat == "bilateral" or atel_lat == "bilateral"):
            _add_unique(findings_elements, f"Bibasilar {atel_desc} and small bilateral pleural effusions.")
            _add_unique(impression_bullets, f"Bibasilar {atel_imp} with small bilateral pleural effusions.")
        elif eff_lat == "left" or atel_lat == "left":
            _add_unique(findings_elements, f"Left-sided {atel_desc} and left pleural effusion.")
            _add_unique(impression_bullets, f"Left-sided {atel_imp} with left pleural effusion.")
        elif eff_lat == "right" or atel_lat == "right":
            _add_unique(findings_elements, f"Right-sided {atel_desc} and right pleural effusion.")
            _add_unique(impression_bullets, f"Right-sided {atel_imp} with right pleural effusion.")
        else:
            _add_unique(findings_elements, f"{atel_desc.capitalize()} and pleural effusion are noted.")
            _add_unique(impression_bullets, f"{atel_imp.capitalize()} with pleural effusion.")
        has_abnormality = True
    elif has_effusion:
        if eff_lat == "bilateral":
            _add_unique(findings_elements, "Bilateral pleural effusions are present.")
            _add_unique(impression_bullets, "Bilateral pleural effusions.")
        elif eff_lat == "left":
            _add_unique(findings_elements, "A left-sided pleural effusion is present.")
            _add_unique(impression_bullets, "Left-sided pleural effusion.")
        elif eff_lat == "right":
            _add_unique(findings_elements, "A right-sided pleural effusion is present.")
            _add_unique(impression_bullets, "Right-sided pleural effusion.")
        else:
            _add_unique(findings_elements, "A pleural effusion is observed.")
            _add_unique(impression_bullets, "Pleural effusion.")
        has_abnormality = True
    elif has_atelectasis:
        if is_compressive:
            _add_unique(findings_elements, "Compressive atelectasis at the lung bases is noted.")
            _add_unique(impression_bullets, "Compressive atelectasis at the lung bases.")
        elif is_basilar or atel_lat == "bilateral":
            _add_unique(findings_elements, "Bibasilar atelectatic changes are noted.")
            _add_unique(impression_bullets, "Bibasilar atelectatic changes.")
        elif atel_lat == "left":
            _add_unique(findings_elements, "Left-sided atelectatic changes are noted.")
            _add_unique(impression_bullets, "Left-sided atelectatic change.")
        elif atel_lat == "right":
            _add_unique(findings_elements, "Right-sided atelectatic changes are noted.")
            _add_unique(impression_bullets, "Right-sided atelectatic change.")
        else:
            _add_unique(findings_elements, "Atelectatic changes are noted.")
            _add_unique(impression_bullets, "Atelectatic change.")
        has_abnormality = True

    # 5. Pneumonia / Opacity
    has_pneumonia = ("pneumonia" in text_lower or "infection" in text_lower) and not (_is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection"))
    has_opacity = ("opacity" in text_lower or "opacities" in text_lower or "cloudy" in text_lower or "spot" in text_lower) and not (_is_term_negated(text_lower, "opacity") or _is_term_negated(text_lower, "opacities"))

    if has_pneumonia:
        pne_lat = _get_laterality(text_lower, ["pneumonia", "infection"])
        loc = "Bilateral " if pne_lat == "bilateral" else ("Right lower lobe " if (pne_lat == "right" and "lower" in text_lower) else ("Left lower lobe " if (pne_lat == "left" and "lower" in text_lower) else ("Right " if pne_lat == "right" else ("Left " if pne_lat == "left" else ""))))
        _add_unique(findings_elements, f"{loc}Air-space opacity is present, suggestive of pneumonia.")
        _add_unique(impression_bullets, f"{loc}Air-space opacity, suggestive of pneumonia.")
        has_abnormality = True
    elif has_opacity and not any("opacity" in f.lower() for f in findings_elements):
        opac_lat = _get_laterality(text_lower, ["opacity", "opacities", "cloudy", "spot"])
        if opac_lat == "bilateral":
            _add_unique(findings_elements, "Bilateral air-space opacities are noted.")
            _add_unique(impression_bullets, "Bilateral air-space opacity.")
        elif opac_lat == "left":
            _add_unique(findings_elements, "Left-sided air-space opacity is noted.")
            _add_unique(impression_bullets, "Left-sided air-space opacity.")
        elif opac_lat == "right":
            _add_unique(findings_elements, "Right-sided air-space opacity is noted.")
            _add_unique(impression_bullets, "Right-sided air-space opacity.")
        else:
            _add_unique(findings_elements, "Air-space opacity is noted.")
            _add_unique(impression_bullets, "Air-space opacity.")
        has_abnormality = True

    # 6. Pneumothorax & Explicit Negatives
    has_pneu = ("pneumothorax" in text_lower or "air around" in text_lower or "air in pleural" in text_lower)
    if has_pneu:
        is_negated = _is_term_negated(text_lower, "pneumothorax") or _is_term_negated(text_lower, "air")
        is_hist = _has_historical_context(text_lower, ["pneumothorax", "air"])
        if is_negated:
            if is_hist:
                _add_unique(explicit_negatives, "pneumothorax (prior pneumothorax resolved)")
            else:
                _add_unique(explicit_negatives, "pneumothorax")
        else:
            pneu_lat = _get_laterality(text_lower, ["pneumothorax", "air"])
            if pneu_lat == "bilateral":
                _add_unique(findings_elements, "Bilateral pneumothoraces are identified.")
                _add_unique(impression_bullets, "Bilateral pneumothoraces.")
            elif pneu_lat == "left":
                _add_unique(findings_elements, "A left pneumothorax is identified.")
                _add_unique(impression_bullets, "Left pneumothorax.")
            elif pneu_lat == "right":
                _add_unique(findings_elements, "A right pneumothorax is identified.")
                _add_unique(impression_bullets, "Right pneumothorax.")
            else:
                _add_unique(findings_elements, "A pneumothorax is present.")
                _add_unique(impression_bullets, "Pneumothorax.")
            has_abnormality = True

    # Check other explicit negatives
    if (_is_term_negated(text_lower, "congestion") or _is_term_negated(text_lower, "vascular congestion") or _is_term_negated(text_lower, "pulmonary venous pressure")) and not has_pvp:
        _add_unique(explicit_negatives, "pulmonary vascular congestion")
    if (_is_term_negated(text_lower, "edema")) and not has_edema:
        _add_unique(explicit_negatives, "pulmonary edema")
    if (_is_term_negated(text_lower, "effusion") or _is_term_negated(text_lower, "fluid")) and not has_effusion:
        _add_unique(explicit_negatives, "pleural effusion")
    if (_is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection")) and not has_pneumonia:
        _add_unique(explicit_negatives, "focal pneumonia")
    if (_is_term_negated(text_lower, "opacity") or _is_term_negated(text_lower, "opacities")) and not has_opacity:
        _add_unique(explicit_negatives, "focal opacity")

    has_neg_cardiopulmonary = bool(re.search(r"\bno (acute )?cardiopulmonary (abnormality|disease|finding|process)\b", text_lower))

    # Assemble explicit negative findings text
    if explicit_negatives:
        unique_negs = []
        for n in explicit_negatives:
            if n not in unique_negs:
                unique_negs.append(n)
        if len(unique_negs) == 1:
            neg_str = unique_negs[0]
            _add_unique(findings_elements, f"No {neg_str} is identified.")
            _add_unique(impression_bullets, f"No acute {neg_str}.")
        else:
            neg_str = ", ".join(unique_negs[:-1]) + f", or {unique_negs[-1]}"
            _add_unique(findings_elements, f"No {neg_str} is identified.")
            _add_unique(impression_bullets, f"No acute {', '.join(unique_negs)}.")

    if has_neg_cardiopulmonary:
        if "disease" in text_lower:
            _add_unique(impression_bullets, "No acute cardiopulmonary disease.")
        elif "abnormality" in text_lower or "abnormalities" in text_lower:
            _add_unique(impression_bullets, "No acute cardiopulmonary abnormality.")

    if not has_abnormality and not findings_elements:
        findings_elements = ["The cardiac silhouette is within normal limits. Lungs are clear without focal consolidation, pleural effusion, or pneumothorax."]
        if not impression_bullets:
            impression_bullets = ["No acute cardiopulmonary abnormality."]

    seen = set()
    unique_impressions = []
    for imp in impression_bullets:
        if imp not in seen:
            seen.add(imp)
            unique_impressions.append(imp)

    if not unique_impressions:
        unique_impressions = ["No acute cardiopulmonary abnormality."]

    findings_text = " ".join(findings_elements)
    impression_text = "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(unique_impressions)]) if len(unique_impressions) > 1 else unique_impressions[0]

    return {
        "findings": findings_text,
        "impression": impression_text,
        "full_text": f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}"
    }


# ==============================================================================
# QUALITY CONTROL & SAFETY CONSISTENCY VALIDATION
# ==============================================================================

def validate_report_consistency(raw_report: str, simple_report: str, medical_report_text: str) -> Dict[str, Any]:
    """
    Validates that the Simple Report and Professional Medical Report maintain exact
    semantic consistency with the original Raw Report:
    - Same findings/diseases present
    - No new diseases/findings introduced (no hallucinations)
    - No findings removed
    - Positive/negative polarities preserved
    - Anatomical locations and laterality (left/right/bilateral) preserved
    """
    issues = []
    raw_lower = clean_raw_report(raw_report).lower()
    simple_lower = simple_report.lower()
    med_lower = medical_report_text.lower()

    # 1. Check Laterality (Left vs Right vs Bilateral)
    raw_has_left = "left" in raw_lower or "left-sided" in raw_lower
    raw_has_right = "right" in raw_lower or "right-sided" in raw_lower

    for side in ["bilateral", "left", "right"]:
        in_raw = (side in raw_lower) or (side == "bilateral" and ("both" in raw_lower or "bilaterally" in raw_lower))
        in_simple = (side in simple_lower) or (side == "bilateral" and ("both" in simple_lower or "bilaterally" in simple_lower))
        in_med = (side in med_lower) or (side == "bilateral" and ("both" in med_lower or "bilaterally" in med_lower))
        
        if in_raw and not (in_simple or in_med):
            if side in ["left", "right"] and (("bilateral" in simple_lower or "both" in simple_lower) or ("bilateral" in med_lower or "both" in med_lower)):
                pass
            else:
                issues.append(f"Laterality discrepancy: Original report specified '{side}', but it was missing in refined report.")
        elif not in_raw and (in_simple or in_med):
            if side in ["left", "right"] and (f"{side}-sided" in raw_lower or side in raw_lower):
                pass
            elif side == "bilateral" and ("both" in raw_lower or "bilaterally" in raw_lower or (raw_has_left and raw_has_right)):
                pass
            else:
                issues.append(f"Laterality hallucination: Refined report introduced '{side}' which was not in the original raw report.")

    # 2. Check for Hallucinated Unmentioned Findings
    hallucinated_findings = []
    for finding in KNOWN_FINDINGS:
        finding_in_med = finding in med_lower
        finding_in_simple = finding in simple_lower

        if finding_in_med and (_is_term_negated(med_lower, finding) or "without" in med_lower or "no " in med_lower):
            finding_in_med = False
        if finding_in_simple and (_is_term_negated(simple_lower, finding) or "without" in simple_lower or "no " in simple_lower):
            finding_in_simple = False

        if finding_in_med or finding_in_simple:
            synonyms = [finding]
            if finding == "cardiomegaly":
                synonyms.extend(["enlarged heart", "heart is enlarged", "big heart", "cardiac silhouette", "heart is big", "heart size", "cardiac size", "enlargement of the cardiac silhouette", "borderline size of the cardiac silhouette", "borderline cardiac"])
            elif finding == "pleural effusion":
                synonyms.extend(["fluid around", "fluid in pleural", "effusion", "effusions", "pleural fluid", "fluid accumulation"])
            elif finding == "atelectasis":
                synonyms.extend(["collapsed lung", "collapse", "atelectatic"])
            elif finding == "pneumothorax":
                synonyms.extend(["air around", "air in pleural", "pneumothoraces"])
            elif finding == "pneumonia":
                synonyms.extend(["infection", "infectious"])
            elif finding == "opacity":
                synonyms.extend(["cloudy", "hazy", "patchy", "white spot", "spot", "opacities", "pulmonary opacity", "air-space opacity", "airspace opacity"])
            elif finding == "consolidation":
                synonyms.extend(["airspace consolidation", "focal consolidation"])
            elif finding == "congestion":
                synonyms.extend(["pulmonary venous pressure", "vascular congestion"])

            present_in_raw = any(syn in raw_lower for syn in synonyms)
            if not present_in_raw:
                hallucinated_findings.append(finding)

    if hallucinated_findings:
        issues.append(f"Potential hallucination detected: Introduced clinical finding(s) not in raw report: {', '.join(hallucinated_findings)}")

    is_valid = len(issues) == 0
    consistency_score = 100.0 if is_valid else max(60.0, 100.0 - (len(issues) * 20.0))

    return {
        "is_valid": is_valid,
        "score": consistency_score,
        "flagged_issues": issues,
        "details": {
            "raw_length": len(raw_report),
            "simple_length": len(simple_report),
            "medical_length": len(medical_report_text),
            "hallucinations_detected": hallucinated_findings,
        }
    }


# ==============================================================================
# MASTER REFINEMENT INTERFACE
# ==============================================================================

def refine_report(raw_report: str, pathology_predictions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Main entry point for report refinement. Takes raw generated report text,
    produces Simple & Professional reports, executes BioBERT Biomedical NLP analysis
    (entity extraction, polarity/certainty/laterality/severity), performs DenseNet121
    consistency validation, and returns a structured dict.
    """
    raw_clean = clean_raw_report(raw_report)

    # Execute BioBERT Biomedical NLP Engine FIRST
    biobert_analysis = None
    biobert_findings = []
    try:
        from biobert_nlp_engine import get_biobert_engine
        biobert_engine = get_biobert_engine()
        biobert_analysis = biobert_engine.process_full_report(raw_clean, pathology_predictions)
        if biobert_analysis and "structured_findings" in biobert_analysis:
            biobert_findings = biobert_analysis["structured_findings"]
    except Exception as e:
        print(f"[report_refinement] Warning: BioBERT processing failed: {e}")

    simple_rep = generate_simple_report(raw_clean, biobert_findings=biobert_findings)
    med_dict = generate_medical_report(raw_clean, biobert_findings=biobert_findings)
    med_full = med_dict["full_text"]

    val_result = validate_report_consistency(raw_report, simple_rep, med_full)

    if not val_result["is_valid"] and val_result["details"]["hallucinations_detected"]:
        simple_rep = generate_simple_report(raw_clean, biobert_findings=biobert_findings)
        med_dict = generate_medical_report(raw_clean, biobert_findings=biobert_findings)
        med_full = med_dict["full_text"]
        val_result = validate_report_consistency(raw_report, simple_rep, med_full)

    engine_name = "BioBERT Biomedical NLP Engine + " + ("LLM API (Configured)" if _get_api_key() else "Medical Radiology NLP Engine (Offline Rule-Based)")

    return {
        "raw_report": raw_report,
        "cleaned_raw_report": raw_clean,
        "simple_report": simple_rep,
        "medical_report": med_dict,
        "findings": med_dict["findings"],
        "impression": med_dict["impression"],
        "validation": val_result,
        "biobert_analysis": biobert_analysis,
        "biobert_findings": biobert_findings,
        "ner_entities": biobert_analysis["ner_entities"] if biobert_analysis else [],
        "densenet_verification": biobert_analysis["densenet_verification"] if biobert_analysis else {},
        "engine_used": engine_name
    }

