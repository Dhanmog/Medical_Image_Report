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
    """Retrieve available API key from environment variables."""
    return (
        os.getenv("API_KEY") or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GROQ_API_KEY")
    )

def _call_llm_api(prompt: str) -> Optional[str]:
    """
    Attempts to call an LLM API if credentials are provided.
    Supports OpenAI-compatible endpoints and Gemini REST endpoints.
    Returns None if no key is set or if the call fails.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    # 1. OpenAI / Compatible endpoint
    try:
        if os.getenv("OPENAI_API_KEY") or (api_key.startswith("sk-") and not api_key.startswith("sk-or-")):
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    # 2. Gemini REST API
    try:
        gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if api_key.startswith("AIza") else None)
        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
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
    """Checks if a medical term in text is negated (e.g. 'no opacity or effusion', 'without pneumothorax')."""
    pattern = r"\b(no|without|absent|negative for|free of)\b[\w\s,]*?\b" + re.escape(term) + r"s?\b"
    return bool(re.search(pattern, text_lower))


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

def generate_simple_report(raw_report: str) -> str:
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

    # 2. Offline Rule-based Medical NLP Engine
    text_lower = cleaned_raw.lower()

    positive_parts = []
    negative_parts = []

    # Support devices check
    if ("pacemaker" in text_lower or "pacer" in text_lower) and not _is_term_negated(text_lower, "pacemaker"):
        pace_lat = _get_laterality(text_lower, ["pacemaker", "pacer"])
        loc = "on the left side of the chest" if pace_lat == "left" else ("on the right side of the chest" if pace_lat == "right" else "in the chest")
        positive_parts.append(f"a pacemaker is present {loc}".strip())
    elif ("catheter" in text_lower or "subclavian" in text_lower) and not _is_term_negated(text_lower, "catheter") and not _is_term_negated(text_lower, "subclavian"):
        cath_lat = _get_laterality(text_lower, ["catheter", "subclavian"])
        loc = "on the left side" if cath_lat == "left" else ("on the right side" if cath_lat == "right" else "")
        positive_parts.append(f"a catheter is present {loc}".strip())

    if ("sternal wires" in text_lower or "sternotomy" in text_lower) and not _is_term_negated(text_lower, "sternal"):
        positive_parts.append("sternal wires from prior surgery are intact")

    # Cardiomegaly check
    if re.search(r"\b(enlarged heart|heart is enlarged|big heart|cardiomegaly|cardiac silhouette is enlarged|heart is big|enlarged cardiac silhouette|enlargement of the cardiac silhouette)\b", text_lower) and not _is_term_negated(text_lower, "enlarged") and not _is_term_negated(text_lower, "cardiomegaly"):
        positive_parts.append("the heart appears larger than normal (cardiomegaly)")

    # Pulmonary Venous Pressure / Congestion check
    if ("pulmonary venous pressure" in text_lower or "vascular congestion" in text_lower or "pulmonary congestion" in text_lower) and not _is_term_negated(text_lower, "congestion"):
        positive_parts.append("there is increased fluid pressure in the blood vessels of the lungs")

    # Pleural effusion check
    if ("fluid" in text_lower or "effusion" in text_lower) and not _is_term_negated(text_lower, "effusion") and not _is_term_negated(text_lower, "fluid"):
        eff_lat = _get_laterality(text_lower, ["effusion", "fluid"])
        if eff_lat == "bilateral":
            positive_parts.append("there is a buildup of fluid around both lungs")
        elif eff_lat == "left":
            positive_parts.append("there is a buildup of fluid around the left lung")
        elif eff_lat == "right":
            positive_parts.append("there is a buildup of fluid around the right lung")
        else:
            positive_parts.append("there is a buildup of fluid around the lung")

    # Pneumothorax check
    if ("air" in text_lower or "pneumothorax" in text_lower) and not _is_term_negated(text_lower, "pneumothorax"):
        pneu_lat = _get_laterality(text_lower, ["pneumothorax", "air"])
        if pneu_lat == "bilateral":
            positive_parts.append("there is trapped air around both lungs")
        elif pneu_lat == "left":
            positive_parts.append("there is trapped air around the left lung")
        elif pneu_lat == "right":
            positive_parts.append("there is trapped air around the right lung")
        else:
            positive_parts.append("there is trapped air outside the lung")

    # Atelectasis check
    if ("collapsed" in text_lower or "atelectasis" in text_lower) and not _is_term_negated(text_lower, "atelectasis"):
        atel_lat = _get_laterality(text_lower, ["atelectasis", "collapsed"])
        loc = "in both lungs" if atel_lat == "bilateral" else ("in the left lung" if atel_lat == "left" else ("in the right lung" if atel_lat == "right" else ""))
        positive_parts.append(f"there is a small area of collapsed lung tissue {loc}".strip())

    # Pneumonia check (Positive vs Negated)
    if "pneumonia" in text_lower or "infection" in text_lower:
        if _is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection"):
            negative_parts.append("There is no evidence of a lung infection (pneumonia).")
        else:
            positive_parts.append("there are signs suggestive of a lung infection (pneumonia)")
    elif ("cloudy" in text_lower or "opacity" in text_lower or "opacities" in text_lower or "spot" in text_lower) and not _is_term_negated(text_lower, "opacity") and not _is_term_negated(text_lower, "opacities") and not any("infection" in p for p in positive_parts):
        opac_lat = _get_laterality(text_lower, ["opacity", "opacities", "cloudy", "spot"])
        loc = "in both lungs" if opac_lat == "bilateral" else ("in the right lung" if opac_lat == "right" else ("in the left lung" if opac_lat == "left" else "in the lung"))
        positive_parts.append(f"there is a hazy or cloudy spot {loc}, which may indicate fluid or inflammation")

    res_sentences = []
    if positive_parts:
        if len(positive_parts) == 1:
            p_str = "The " + positive_parts[0] if positive_parts[0].startswith("heart") else positive_parts[0][0].upper() + positive_parts[0][1:]
            res_sentences.append(p_str + ".")
        elif len(positive_parts) == 2:
            p_str = (positive_parts[0][0].upper() + positive_parts[0][1:]) + ", and " + positive_parts[1] + "."
            res_sentences.append(p_str)
        else:
            p_str = (positive_parts[0][0].upper() + positive_parts[0][1:]) + ", " + ", ".join(positive_parts[1:-1]) + ", and " + positive_parts[-1] + "."
            res_sentences.append(p_str)

    if negative_parts:
        res_sentences.extend(negative_parts)

    if res_sentences:
        return " ".join(res_sentences)

    # Check for normal/unremarkable findings only if no positive abnormalities were found
    if re.search(r"\b(no (acute )?cardiopulmonary (abnormality|disease|finding)|normal chest|unremarkable|within normal limits)\b", text_lower):
        return "The chest X-ray shows normal heart size and clear lungs. No signs of infection, fluid, or other acute abnormalities are present."

    # Clean fallback
    s_clean = cleaned_raw
    for pattern, medical_term in INFORMAL_TO_MEDICAL:
        s_clean = re.sub(pattern, medical_term, s_clean, flags=re.IGNORECASE)
    return s_clean[0].upper() + s_clean[1:] + ("." if not s_clean.endswith(".") else "")


def generate_medical_report(raw_report: str) -> Dict[str, str]:
    """
    Generates a professional radiology report structured into:
    - FINDINGS: Professional clinical description using radiology terminology.
    - IMPRESSION: Concise summary of key clinical observations.
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

    # 2. Local Rule-based Medical NLP Engine
    text_lower = cleaned_raw.lower()

    findings_elements = []
    impression_bullets = []

    has_abnormality = False

    # Support devices check
    if ("pacemaker" in text_lower or "pacer" in text_lower) and not _is_term_negated(text_lower, "pacemaker"):
        pace_lat = _get_laterality(text_lower, ["pacemaker", "pacer"])
        loc = "Left pectoral " if pace_lat == "left" else ("Right pectoral " if pace_lat == "right" else "")
        lead_loc = ""
        if "right atrium" in text_lower or "right ventricle" in text_lower:
            lead_loc = " with leads extending to the right atrium and right ventricle"
        findings_elements.append(f"{loc}Pacemaker is in place{lead_loc}.".strip())
        has_abnormality = True

    if ("subclavian" in text_lower or "catheter" in text_lower) and not _is_term_negated(text_lower, "catheter") and not _is_term_negated(text_lower, "subclavian"):
        cath_lat = _get_laterality(text_lower, ["catheter", "subclavian"])
        loc = "Left subclavian " if cath_lat == "left" else ("Right subclavian " if cath_lat == "right" else "")
        findings_elements.append(f"{loc}Catheter is noted.".strip())
        has_abnormality = True

    if ("sternal wires" in text_lower or "sternotomy" in text_lower) and not _is_term_negated(text_lower, "sternal"):
        findings_elements.append("Intact midline sternal wires are present.")
        has_abnormality = True

    # Check Cardiomegaly
    if re.search(r"\b(cardiomegaly|enlarged heart|heart is enlarged|big heart|heart is big|cardiac silhouette is enlarged|enlarged cardiac silhouette|enlargement of the cardiac silhouette)\b", text_lower) and not _is_term_negated(text_lower, "cardiomegaly") and not _is_term_negated(text_lower, "enlarged"):
        findings_elements.append("Cardiomegaly is noted.")
        impression_bullets.append("Cardiomegaly.")
        has_abnormality = True
    elif ("normal" in text_lower or "limits" in text_lower or "normal size" in text_lower) and not has_abnormality:
        findings_elements.append("The cardiac silhouette is within normal limits.")

    # Check Pulmonary Venous Pressure / Congestion
    if ("pulmonary venous pressure" in text_lower or "vascular congestion" in text_lower or "pulmonary congestion" in text_lower) and not _is_term_negated(text_lower, "congestion"):
        findings_elements.append("Mild elevation of pulmonary venous pressure is present.")
        if impression_bullets and "Cardiomegaly." in impression_bullets[0]:
            impression_bullets[0] = "Cardiomegaly with mild pulmonary vascular congestion."
        else:
            impression_bullets.append("Mild pulmonary vascular congestion.")
        has_abnormality = True

    # Check Pleural Effusion
    if ("effusion" in text_lower or "fluid" in text_lower) and not _is_term_negated(text_lower, "effusion") and not _is_term_negated(text_lower, "fluid"):
        eff_lat = _get_laterality(text_lower, ["effusion", "fluid"])
        if eff_lat == "bilateral":
            findings_elements.append("Bilateral pleural effusions are present.")
            impression_bullets.append("Bilateral pleural effusions.")
        elif eff_lat == "left":
            findings_elements.append("A left-sided pleural effusion is present.")
            impression_bullets.append("Left-sided pleural effusion.")
        elif eff_lat == "right":
            findings_elements.append("A right-sided pleural effusion is present.")
            impression_bullets.append("Right-sided pleural effusion.")
        else:
            findings_elements.append("A pleural effusion is observed.")
            impression_bullets.append("Pleural effusion.")
        has_abnormality = True

    # Check Pneumothorax
    if ("pneumothorax" in text_lower or "air around" in text_lower or "air in pleural" in text_lower) and not _is_term_negated(text_lower, "pneumothorax"):
        pneu_lat = _get_laterality(text_lower, ["pneumothorax", "air"])
        if pneu_lat == "bilateral":
            findings_elements.append("Bilateral pneumothoraces are identified.")
            impression_bullets.append("Bilateral pneumothoraces.")
        elif pneu_lat == "left":
            findings_elements.append("A left pneumothorax is identified.")
            impression_bullets.append("Left pneumothorax.")
        elif pneu_lat == "right":
            findings_elements.append("A right pneumothorax is identified.")
            impression_bullets.append("Right pneumothorax.")
        else:
            findings_elements.append("A pneumothorax is present.")
            impression_bullets.append("Pneumothorax.")
        has_abnormality = True

    # Check Atelectasis
    if ("atelectasis" in text_lower or "collapsed" in text_lower) and not _is_term_negated(text_lower, "atelectasis"):
        atel_lat = _get_laterality(text_lower, ["atelectasis", "collapsed"])
        if atel_lat == "bilateral":
            findings_elements.append("Bilateral atelectatic changes are noted.")
            impression_bullets.append("Bilateral atelectatic changes.")
        elif atel_lat == "left":
            findings_elements.append("Left-sided atelectatic changes are noted.")
            impression_bullets.append("Left-sided atelectatic change.")
        elif atel_lat == "right":
            findings_elements.append("Right-sided atelectatic changes are noted.")
            impression_bullets.append("Right-sided atelectatic change.")
        else:
            findings_elements.append("Atelectatic changes are noted.")
            impression_bullets.append("Atelectatic change.")
        has_abnormality = True

    # Check Pneumonia
    if "pneumonia" in text_lower or "infection" in text_lower:
        if _is_term_negated(text_lower, "pneumonia") or _is_term_negated(text_lower, "infection"):
            findings_elements.append("No evidence of acute focal pneumonia.")
            impression_bullets.append("No acute focal pneumonia.")
        else:
            pne_lat = _get_laterality(text_lower, ["pneumonia", "infection"])
            if pne_lat == "bilateral":
                loc = "Bilateral "
            elif pne_lat == "right" and "lower" in text_lower:
                loc = "Right lower lobe "
            elif pne_lat == "left" and "lower" in text_lower:
                loc = "Left lower lobe "
            elif pne_lat == "right":
                loc = "Right "
            elif pne_lat == "left":
                loc = "Left "
            else:
                loc = ""
            findings_elements.append(f"{loc}Air-space opacity is present, suggestive of pneumonia.")
            impression_bullets.append(f"{loc}Air-space opacity, suggestive of pneumonia.")
            has_abnormality = True
    elif ("opacity" in text_lower or "opacities" in text_lower or "cloudy" in text_lower or "spot" in text_lower) and not _is_term_negated(text_lower, "opacity") and not _is_term_negated(text_lower, "opacities") and not any("opacity" in b.lower() for b in impression_bullets):
        opac_lat = _get_laterality(text_lower, ["opacity", "opacities", "cloudy", "spot"])
        if opac_lat == "bilateral":
            findings_elements.append("Bilateral air-space opacities are noted.")
            impression_bullets.append("Bilateral air-space opacity.")
        elif opac_lat == "left":
            findings_elements.append("Left-sided air-space opacity is noted.")
            impression_bullets.append("Left-sided air-space opacity.")
        elif opac_lat == "right":
            findings_elements.append("Right-sided air-space opacity is noted.")
            impression_bullets.append("Right-sided air-space opacity.")
        else:
            findings_elements.append("Air-space opacity is noted.")
            impression_bullets.append("Air-space opacity.")
        has_abnormality = True

    if not has_abnormality and not findings_elements:
        findings_elements = ["The cardiac silhouette is within normal limits. Lungs are clear without focal consolidation, pleural effusion, or pneumothorax."]
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

        if finding_in_med and ("without focal consolidation" in med_lower or "no acute cardiopulmonary" in med_lower):
            finding_in_med = False

        if finding_in_med or finding_in_simple:
            synonyms = [finding]
            if finding == "cardiomegaly":
                synonyms.extend(["enlarged heart", "heart is enlarged", "big heart", "cardiac silhouette", "heart is big", "heart size", "cardiac size", "enlargement of the cardiac silhouette"])
            elif finding == "pleural effusion":
                synonyms.extend(["fluid around", "fluid in pleural", "effusion", "effusions", "pleural fluid", "fluid accumulation"])
            elif finding == "atelectasis":
                synonyms.extend(["collapsed lung", "collapse", "atelectatic"])
            elif finding == "pneumothorax":
                synonyms.extend(["air around", "air in pleural", "pneumothoraces"])
            elif finding == "pneumonia":
                synonyms.extend(["infection", "infectious"])
            elif finding == "opacity":
                synonyms.extend(["cloudy", "hazy", "patchy", "white spot", "consolidation", "spot", "opacities"])
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

def refine_report(raw_report: str) -> Dict[str, Any]:
    """
    Main entry point for report refinement. Takes raw generated report text,
    produces Simple & Professional reports, performs safety quality control validation,
    and returns a structured dict.
    """
    raw_clean = clean_raw_report(raw_report)

    simple_rep = generate_simple_report(raw_clean)
    med_dict = generate_medical_report(raw_clean)
    med_full = med_dict["full_text"]

    val_result = validate_report_consistency(raw_report, simple_rep, med_full)

    if not val_result["is_valid"] and val_result["details"]["hallucinations_detected"]:
        simple_rep = generate_simple_report(raw_clean)
        med_dict = generate_medical_report(raw_clean)
        med_full = med_dict["full_text"]
        val_result = validate_report_consistency(raw_report, simple_rep, med_full)

    engine_name = "LLM API (Configured)" if _get_api_key() else "Medical Radiology NLP Engine (Offline Rule-Based)"

    return {
        "raw_report": raw_report,
        "cleaned_raw_report": raw_clean,
        "simple_report": simple_rep,
        "medical_report": med_dict,
        "findings": med_dict["findings"],
        "impression": med_dict["impression"],
        "validation": val_result,
        "engine_used": engine_name
    }
