"""
================================================================================
TEST SUITE FOR MEDICAL CONCEPT-BASED MULTI-VIEW REPORT CONSOLIDATION (PRIORITY 2)
================================================================================
Tests all required multi-view consolidation scenarios:
  1. Two-view input with synonymous findings (e.g. Cardiomegaly vs Enlarged Cardiac Silhouette)
  2. Three-view input with complementary findings
  3. Duplicate findings deduplication
  4. Contradictory findings preservation (Positive vs Negative)
  5. Different anatomical locations preservation (Left vs Right)
  6. Empty / Invalid reports handling
  7. Full end-to-end integration with report_refinement.py
================================================================================
"""

import sys
import os
import re

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from report_refinement import refine_report, clean_raw_report


def extract_clinical_concept(sentence: str) -> dict:
    """
    Parses a clinical sentence into a structured concept dictionary:
    {
        "class": pathology_class,
        "laterality": LEFT / RIGHT / BILATERAL / UNSPECIFIED,
        "polarity": POSITIVE / NEGATIVE,
        "severity": MILD / MODERATE / SEVERE / UNSPECIFIED,
        "original_text": sentence
    }
    """
    text_lower = sentence.lower().strip()

    # 1. Detect Polarity (Negation check)
    is_negated = bool(re.search(r"\b(no|without|absent|negative for|free of|denies)\b", text_lower))
    polarity = "NEGATIVE" if is_negated else "POSITIVE"

    # 2. Detect Laterality
    if "bilateral" in text_lower or "both" in text_lower:
        laterality = "BILATERAL"
    elif "left" in text_lower:
        laterality = "LEFT"
    elif "right" in text_lower:
        laterality = "RIGHT"
    else:
        laterality = "UNSPECIFIED"

    # 3. Detect Severity
    if re.search(r"\b(mild|slight|small)\b", text_lower):
        severity = "MILD"
    elif re.search(r"\b(severe|marked|large|massive)\b", text_lower):
        severity = "SEVERE"
    elif "moderate" in text_lower:
        severity = "MODERATE"
    else:
        severity = "UNSPECIFIED"

    # 4. Detect Pathology Class
    if re.search(r"\b(cardiomegaly|enlarged heart|heart is enlarged|big heart|heart is big|cardiac silhouette is enlarged|enlarged cardiac silhouette|enlargement of the cardiac silhouette|cardiac size is enlarged)\b", text_lower):
        path_class = "CARDIOMEGALY"
    elif re.search(r"\b(pleural effusion|effusion|fluid around|pleural fluid)\b", text_lower):
        path_class = "PLEURAL_EFFUSION"
    elif re.search(r"\b(pneumothorax|air around|air in pleural)\b", text_lower):
        path_class = "PNEUMOTHORAX"
    elif re.search(r"\b(atelectasis|collapsed lung|lung collapse|atelectatic)\b", text_lower):
        path_class = "ATELECTASIS"
    elif re.search(r"\b(pneumonia|lung infection|infection in lung|focal pneumonia)\b", text_lower):
        path_class = "PNEUMONIA"
    elif re.search(r"\b(pulmonary edema|edema|vascular congestion|pulmonary congestion|pulmonary venous pressure)\b", text_lower):
        path_class = "EDEMA"
    elif re.search(r"\b(opacity|opacities|cloudy|hazy|infiltrate|consolidation)\b", text_lower):
        path_class = "OPACITY"
    elif re.search(r"\b(normal limits|unremarkable|clear lungs|normal heart|no acute|normal cardiac)\b", text_lower):
        path_class = "NORMAL"
    else:
        path_class = f"GENERIC_{text_lower}"

    return {
        "class": path_class,
        "laterality": laterality,
        "polarity": polarity,
        "severity": severity,
        "original_text": sentence
    }


def consolidate_multi_view_reports(raw_reports: list) -> str:
    """
    Consolidates raw diagnostic text observations from 1 to 3 chest X-ray views using conservative
    medical concept-based deduplication.
    """
    if not raw_reports:
        return "No diagnostic findings generated."
    
    cleaned_reports = []
    for r in raw_reports:
        if not r or not isinstance(r, str):
            continue
        cleaned = clean_raw_report(r) if clean_raw_report is not None else r.strip()
        if cleaned and cleaned != "No report generated.":
            cleaned_reports.append(cleaned)
            
    if not cleaned_reports:
        return "No diagnostic findings generated."
    if len(cleaned_reports) == 1:
        return cleaned_reports[0]

    concept_map = {}
    ordered_concepts = []

    for view_idx, report in enumerate(cleaned_reports):
        sentences = [s.strip() for s in re.split(r'[.;]\s*', report) if s.strip()]
        
        for sentence in sentences:
            concept = extract_clinical_concept(sentence)
            key = (concept["class"], concept["laterality"], concept["polarity"], concept["severity"])
            
            if key not in concept_map:
                concept_map[key] = {
                    "representative_text": sentence,
                    "views": [view_idx + 1],
                    "concept": concept
                }
                ordered_concepts.append(key)
            else:
                concept_map[key]["views"].append(view_idx + 1)

    if not ordered_concepts:
        return "The chest X-ray views are unremarkable."

    consolidated_sentences = [concept_map[k]["representative_text"] for k in ordered_concepts]
    
    consolidated_text = ". ".join(consolidated_sentences)
    if not consolidated_text.endswith("."):
        consolidated_text += "."
    return consolidated_text


CONSOLIDATION_TEST_CASES = [
    {
        "id": 1,
        "name": "Two-View Input (Synonymous Cardiomegaly Findings)",
        "views": [
            "Cardiomegaly is noted. Right pleural effusion is present.",
            "The cardiac silhouette is enlarged. Right pleural effusion is present."
        ],
        "expected_terms": ["cardiomegaly", "pleural effusion"],
        "forbidden_terms": ["cardiac silhouette is enlarged"]
    },
    {
        "id": 2,
        "name": "Three-View Input (Complementary Findings Across Views)",
        "views": [
            "Cardiomegaly is noted.",
            "Right pleural effusion is present.",
            "Atelectasis in left lower lung."
        ],
        "expected_terms": ["cardiomegaly", "pleural effusion", "atelectasis"],
        "forbidden_terms": []
    },
    {
        "id": 3,
        "name": "Duplicate Exact Sentences Across 3 Views",
        "views": [
            "The heart size is within normal limits. Clear lungs.",
            "The heart size is within normal limits. Clear lungs.",
            "The heart size is within normal limits. Clear lungs."
        ],
        "expected_terms": ["normal limits"],
        "forbidden_terms": []
    },
    {
        "id": 4,
        "name": "Different Anatomical Locations (Left vs Right Effusion)",
        "views": [
            "Left pleural effusion is present.",
            "Right pleural effusion is present."
        ],
        "expected_terms": ["left", "right", "effusion"],
        "forbidden_terms": []
    },
    {
        "id": 5,
        "name": "Contradictory Findings (Positive vs Negative)",
        "views": [
            "Right pneumothorax is identified.",
            "No right pneumothorax."
        ],
        "expected_terms": ["pneumothorax", "no right pneumothorax"],
        "forbidden_terms": []
    },
    {
        "id": 6,
        "name": "Empty or Noise Input Views",
        "views": [
            "in comparison with",
            ""
        ],
        "expected_terms": [],
        "forbidden_terms": []
    }
]


def run_consolidation_tests():
    print("=" * 85)
    print("      MEDICAL CONCEPT MULTI-VIEW CONSOLIDATION TEST SUITE (PRIORITY 2)")
    print("=" * 85)

    passed_count = 0
    total_count = len(CONSOLIDATION_TEST_CASES)

    for case in CONSOLIDATION_TEST_CASES:
        print(f"\n--- TEST #{case['id']}: {case['name']} ---")
        print("INPUT VIEWS:")
        for idx, view_text in enumerate(case["views"]):
            print(f"  • View {idx+1}: \"{view_text}\"")

        consolidated_raw = consolidate_multi_view_reports(case["views"])
        print(f"CONSOLIDATED RAW  : \"{consolidated_raw}\"")

        refined = refine_report(consolidated_raw)
        print(f"SIMPLE REPORT     : \"{refined['simple_report']}\"")
        print(f"PROFESSIONAL      :\n{refined['medical_report']['full_text']}")

        # Assertion checks
        test_passed = True
        
        # Verify expected terms
        all_text = (consolidated_raw + " " + refined['simple_report'] + " " + refined['medical_report']['full_text']).lower()
        for term in case["expected_terms"]:
            if term.lower() not in all_text:
                print(f"  ❌ FAILED: Expected term '{term}' not found in consolidated report!")
                test_passed = False

        # Verify forbidden (duplicated) terms
        for term in case["forbidden_terms"]:
            if term.lower() in consolidated_raw.lower():
                print(f"  ❌ FAILED: Synonymous duplicate term '{term}' was NOT deduplicated!")
                test_passed = False

        if test_passed:
            print("STATUS            : [PASSED]")
            passed_count += 1
        else:
            print("STATUS            : [FAILED]")

    print("\n" + "=" * 85)
    print(f"SUMMARY: {passed_count}/{total_count} Consolidation Tests Passed ({passed_count/total_count * 100:.1f}%)")
    print("=" * 85)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_consolidation_tests()
    sys.exit(0 if success else 1)
