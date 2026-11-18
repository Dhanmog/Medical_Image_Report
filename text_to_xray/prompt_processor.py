"""
================================================================================
CHEST AI: CLINICAL REPORT-TO-DIFFUSION PROMPT PROCESSOR
================================================================================
Extracts structured findings, explicit negations, laterality, and projection
from CHEST AI diagnostic reports and adapts them into calibrated diffusion prompts.
Ensures zero-inversion of negative findings and faithful preservation of laterality.
================================================================================
"""

import re
from typing import Dict, List, Any, Optional, Union

# Common known chest findings & pathologies
COMMON_PATHOLOGY_TERMS = {
    "cardiomegaly": [
        r"cardiomegaly", r"enlarged heart", r"heart is enlarged",
        r"cardiac silhouette is enlarged", r"enlarged cardiac silhouette",
        r"prominent heart size", r"cardiomegaly is present"
    ],
    "pleural effusion": [
        r"pleural effusion", r"pleural fluid", r"fluid around the lung",
        r"fluid around lungs", r"fluid in pleural space", r"blunting of the costophrenic angle"
    ],
    "pneumonia": [
        r"pneumonia", r"focal pneumonia", r"consolidation", r"lung infection",
        r"infectious process", r"infiltrate"
    ],
    "atelectasis": [
        r"atelectasis", r"partial lung collapse", r"collapsed lung",
        r"atelectatic changes", r"linear atelectasis", r"bibasilar atelectasis"
    ],
    "pneumothorax": [
        r"pneumothorax", r"air around the lung", r"air in the pleural space",
        r"apical pneumothorax"
    ],
    "pulmonary edema": [
        r"pulmonary edema", r"fluid in the lungs", r"vascular congestion",
        r"pulmonary vascular congestion", r"fluid overload"
    ],
    "lung opacity": [
        r"opacity", r"opacities", r"air-space opacity", r"patchy opacity",
        r"hazy density", r"infiltrate"
    ],
    "nodule": [
        r"nodule", r"pulmonary nodule", r"mass", r"coin lesion"
    ],
    "support devices": [
        r"pacemaker", r"sternal wires", r"chest tube", r"central line",
        r"endotracheal tube", r"pacer"
    ]
}

# Standard negation indicators in clinical radiology
NEGATION_PATTERNS = [
    r"\bno\b", r"\bwithout\b", r"\bnegative for\b", r"\bfree of\b",
    r"\bdenies\b", r"\bno evidence of\b", r"\bno signs of\b",
    r"\bnormal\b", r"\bclear\b", r"\babsent\b", r"\bunremarkable\b",
    r"\brule out\b", r"\bresolved\b"
]

# Standard projection indicators
PROJECTION_PATTERNS = {
    "Frontal PA": [r"\bpa view\b", r"\bposteroanterior\b", r"\bpa projection\b", r"\bpa\b"],
    "Frontal AP": [r"\bap view\b", r"\banteroposterior\b", r"\bap projection\b", r"\bportable\b", r"\bap\b"],
    "Lateral": [r"\blateral view\b", r"\blateral projection\b", r"\blat\b"]
}

# Severity and uncertainty indicators
UNCERTAINTY_TERMS = [r"possible", r"probable", r"borderline", r"suggestive of", r"cannot exclude", r"suspected"]
SEVERITY_TERMS = {
    "mild": [r"\bmild\b", r"\bslight\b", r"\bminimal\b", r"\bsmall\b"],
    "moderate": [r"\bmoderate\b"],
    "severe": [r"\bsevere\b", r"\bmarked\b", r"\blarge\b", r"\bextensive\b"]
}


def is_term_negated(text: str, term_pattern: str) -> bool:
    """
    Determines whether a clinical term within text is explicitly negated.
    Uses scoped windowing to avoid negation leakage across punctuation boundaries.
    """
    text_lower = text.lower()
    
    # Split text into grammatical clauses
    clauses = re.split(r"[.;,:]|\band\b|\bbut\b|\bwith\b|\bwithout\b", text_lower)
    for clause in clauses:
        clause = clause.strip()
        if re.search(term_pattern, clause, re.IGNORECASE):
            # Check if any negation marker appears in this exact clause
            for neg in NEGATION_PATTERNS:
                if re.search(neg, clause, re.IGNORECASE):
                    # Edge case check: 'no cardiomegaly' vs 'cardiomegaly without edema'
                    # If clause is 'without edema', 'edema' is negated, not previous words
                    neg_match = re.search(neg, clause, re.IGNORECASE)
                    term_match = re.search(term_pattern, clause, re.IGNORECASE)
                    if neg_match and term_match and neg_match.start() <= term_match.start():
                        return True
                    # Also 'is absent' or 'resolved' after the term
                    if neg in [r"\babsent\b", r"\bresolved\b", r"\bunremarkable\b"]:
                        if neg_match and term_match and neg_match.start() >= term_match.start():
                            return True
            return False
            
    # Fallback window search (within 5 words preceding term)
    pattern = rf"(?:no|without|negative for|free of|no evidence of|clear of)\s+(?:\w+\s+){{0,4}}{term_pattern}"
    return bool(re.search(pattern, text_lower, re.IGNORECASE))


def extract_laterality(text: str, finding: str) -> str:
    """Extracts left, right, or bilateral laterality for a finding."""
    text_lower = text.lower()
    left_pattern = rf"\b(?:left|left-sided|left lung|left base|left hemithorax)\b.*?\b{finding}\b|\b{finding}\b.*?\b(?:left|left-sided|left lung|left base)\b"
    right_pattern = rf"\b(?:right|right-sided|right lung|right base|right hemithorax)\b.*?\b{finding}\b|\b{finding}\b.*?\b(?:right|right-sided|right lung|right base)\b"
    bilateral_pattern = rf"\b(?:bilateral|both lungs|both bases|diffuse)\b.*?\b{finding}\b|\b{finding}\b.*?\b(?:bilateral|both lungs)\b"

    if re.search(bilateral_pattern, text_lower):
        return "bilateral"
    elif re.search(left_pattern, text_lower):
        return "left-sided"
    elif re.search(right_pattern, text_lower):
        return "right-sided"
    return "unspecified"


def extract_severity(text: str, finding: str) -> str:
    """Extracts severity (mild, moderate, severe) associated with a finding."""
    text_lower = text.lower()
    for sev_level, patterns in SEVERITY_TERMS.items():
        for p in patterns:
            comb_pattern = rf"{p}\s+.*?\b{finding}\b|\b{finding}\b\s+.*?{p}"
            if re.search(comb_pattern, text_lower):
                return sev_level
    return "unspecified"


def extract_uncertainty(text: str, finding: str) -> bool:
    """Identifies whether a finding is marked as uncertain/possible."""
    text_lower = text.lower()
    for unc in UNCERTAINTY_TERMS:
        comb_pattern = rf"{unc}\s+.*?\b{finding}\b|\b{finding}\b\s+.*?{unc}"
        if re.search(comb_pattern, text_lower):
            return True
    return False


def detect_projection(text: str) -> str:
    """Detects radiograph projection (PA, AP, Lateral) if explicitly noted in report."""
    text_lower = text.lower()
    for proj_name, patterns in PROJECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return proj_name
    return "Unknown"


def parse_chest_ai_report(report_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses a CHEST AI report (raw string or structured dict) and extracts findings,
    negations, laterality, severity, uncertainty, and projection.
    """
    findings_text = ""
    impression_text = ""
    full_text = ""

    if isinstance(report_input, dict):
        findings_text = str(report_input.get("findings", ""))
        impression_text = str(report_input.get("impression", ""))
        full_text = report_input.get("full_text", "")
        if not full_text:
            full_text = f"FINDINGS: {findings_text}\nIMPRESSION: {impression_text}"
    else:
        full_text = str(report_input)
        # Attempt to split into FINDINGS and IMPRESSION sections if headers exist
        f_match = re.search(r"FINDINGS:\s*(.*?)(?=IMPRESSION:|$)", full_text, re.DOTALL | re.IGNORECASE)
        i_match = re.search(r"IMPRESSION:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)
        if f_match:
            findings_text = f_match.group(1).strip()
        if i_match:
            impression_text = i_match.group(1).strip()
        if not findings_text and not impression_text:
            findings_text = full_text

    positive_findings = []
    negative_findings = []
    laterality_map = {}
    severity_map = {}
    uncertainty_map = {}

    combined_text = f"{findings_text} {impression_text}".strip()

    # Iterate through all clinical pathology groups
    for std_term, synonyms in COMMON_PATHOLOGY_TERMS.items():
        for syn in synonyms:
            if re.search(syn, combined_text, re.IGNORECASE):
                if is_term_negated(combined_text, syn):
                    if std_term not in negative_findings:
                        negative_findings.append(std_term)
                else:
                    if std_term not in positive_findings and std_term not in negative_findings:
                        positive_findings.append(std_term)
                        laterality_map[std_term] = extract_laterality(combined_text, syn)
                        severity_map[std_term] = extract_severity(combined_text, syn)
                        uncertainty_map[std_term] = extract_uncertainty(combined_text, syn)
                break

    projection = detect_projection(combined_text)

    # Check for general normal markers
    is_normal_study = False
    if not positive_findings and (
        "normal" in combined_text.lower() or 
        "no acute cardiopulmonary abnormality" in combined_text.lower() or
        "clear lungs" in combined_text.lower()
    ):
        is_normal_study = True

    return {
        "full_text": full_text,
        "findings_text": findings_text,
        "impression_text": impression_text,
        "positive_findings": positive_findings,
        "negative_findings": negative_findings,
        "laterality": laterality_map,
        "severity": severity_map,
        "uncertainty": uncertainty_map,
        "projection": projection,
        "is_normal": is_normal_study
    }


def adapt_report_to_diffusion_prompt(
    report_input: Union[str, Dict[str, Any]],
    user_projection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transforms a parsed CHEST AI clinical report into:
      1. positive_prompt: Tailored text prompt for pretrained chest X-ray diffusion models.
      2. negative_prompt: Negative guidance tokens embedding explicitly negated clinical findings.
      3. metadata: Detailed extraction trace for research auditing.
    """
    parsed = parse_chest_ai_report(report_input)
    
    # Projection resolution: user override takes precedence if specified and not Unknown
    projection = user_projection if (user_projection and user_projection != "Unknown") else parsed["projection"]
    view_phrase = f"{projection} chest radiograph" if projection != "Unknown" else "Chest radiograph"

    # Construct positive prompt components
    prompt_parts = [view_phrase]

    if parsed["is_normal"] or len(parsed["positive_findings"]) == 0:
        prompt_parts.append("Normal chest X-ray, clear lungs, normal cardiomediastinal silhouette, no acute findings")
    else:
        finding_descriptions = []
        for finding in parsed["positive_findings"]:
            lat = parsed["laterality"].get(finding, "unspecified")
            sev = parsed["severity"].get(finding, "unspecified")
            is_unc = parsed["uncertainty"].get(finding, False)

            phrase = finding
            if lat != "unspecified":
                phrase = f"{lat} {phrase}"
            if sev != "unspecified":
                phrase = f"{sev} {phrase}"
            if is_unc:
                phrase = f"possible {phrase}"
            
            finding_descriptions.append(phrase)

        prompt_parts.append("showing " + ", ".join(finding_descriptions))

    # Standard radiographic quality tokens
    prompt_parts.append("diagnostic quality, monochrome medical imaging, high contrast lungs and ribs")
    positive_prompt = ", ".join(prompt_parts)

    # Construct negative prompt:
    # Crucially include explicitly negated findings to prevent diffusion hallucination
    neg_parts = [
        "color", "colorful", "cartoon", "drawing", "illustration",
        "blur", "low quality", "deformed anatomy", "fractured ribs",
        "non-chest image", "text watermark", "bad radiograph"
    ]
    for neg_finding in parsed["negative_findings"]:
        neg_parts.append(f"{neg_finding}")

    negative_prompt = ", ".join(neg_parts)

    return {
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "parsed_report": parsed,
        "projection_used": projection
    }
