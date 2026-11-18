"""
================================================================================
CLINICAL TEST SUITE FOR DUAL REPORT GENERATION & REFINEMENT MODULE
================================================================================
Tests all 10 required clinical report scenarios to verify:
  1. Raw Report -> Patient-Friendly Simple Report
  2. Raw Report -> Professional Radiology Report (FINDINGS + IMPRESSION)
  3. Medical Terminology Conversion
  4. Semantic Consistency & Quality Control Validation (No hallucinations, laterality preserved)
================================================================================
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from report_refinement import (
    refine_report,
    generate_simple_report,
    generate_medical_report,
    validate_report_consistency
)

TEST_CASES = [
    {
        "id": 1,
        "name": "Normal Chest X-ray",
        "raw": "The cardiac silhouette and lung fields are within normal limits. No focal opacity or pleural effusion."
    },
    {
        "id": 2,
        "name": "Pneumonia",
        "raw": "Right lower lung infection with patchy cloudy area suggestive of pneumonia."
    },
    {
        "id": 3,
        "name": "Cardiomegaly",
        "raw": "The cardiac silhouette is enlarged."
    },
    {
        "id": 4,
        "name": "Pleural Effusion",
        "raw": "The cardiac silhouette is normal size with fluid around the left lung."
    },
    {
        "id": 5,
        "name": "Pneumothorax",
        "raw": "Air around the right lung present."
    },
    {
        "id": 6,
        "name": "Multiple Findings",
        "raw": "The heart is enlarged with fluid around the left lung and a collapsed lung."
    },
    {
        "id": 7,
        "name": "No Abnormality Detected",
        "raw": "No acute cardiopulmonary abnormality."
    },
    {
        "id": 8,
        "name": "Left/Right-Specific Findings",
        "raw": "Right lower lung cloudy spot with left pleural fluid."
    },
    {
        "id": 9,
        "name": "Poor Grammar Raw Report",
        "raw": "the heart is big and left lung have fluid accumulation."
    },
    {
        "id": 10,
        "name": "Incomplete Sentences Raw Report",
        "raw": "enlarged heart left pleural effusion lower lobe opacity"
    },
    {
        "id": 11,
        "name": "Dangling Comparison Preamble",
        "raw": "in comparison with the study of cardiomegaly is noted."
    },
    {
        "id": 12,
        "name": "Repeated Inline Section Headers",
        "raw": "findings: impression: cardiomegaly is noted. impression: no pleural effusion."
    },
    {
        "id": 13,
        "name": "Empty Input Stream",
        "raw": ""
    }
]


def run_tests():
    print("=" * 80)
    print("      CHEST X-RAY REPORT REFINEMENT MODULE - TEST SUITE")
    print("=" * 80)
    
    # Test clean_raw_report directly first
    from report_refinement import clean_raw_report
    print("\n--- DIRECT TEST: clean_raw_report() ---")
    dangling_test = clean_raw_report("findings: impression: in comparison with the study of cardiomegaly is noted.")
    print(f"Dangling Preamble Cleaned Output : \"{dangling_test}\"")
    assert "in comparison" not in dangling_test.lower(), "Dangling preamble failed to clean!"
    assert "findings" not in dangling_test.lower(), "Repeated header failed to clean!"

    empty_test = clean_raw_report("")
    print(f"Empty Input Cleaned Output       : \"{empty_test}\"")
    assert empty_test == "", "Empty input cleaning failed!"
    print("Direct clean_raw_report() assertion checks passed!\n")
    
    passed_count = 0
    total_count = len(TEST_CASES)

    for case in TEST_CASES:
        print(f"\n--- TEST #{case['id']}: {case['name']} ---")
        raw = case["raw"]
        print(f"RAW REPORT     : \"{raw}\"")
        
        result = refine_report(raw)
        
        simple = result["simple_report"]
        med = result["medical_report"]
        val = result["validation"]
        
        print(f"SIMPLE REPORT  : \"{simple}\"")
        print(f"PROFESSIONAL   :\n{med['full_text']}")
        print(f"VALIDATION     : Score={val['score']}% | Valid={val['is_valid']}")
        
        if val["flagged_issues"]:
            print(f"FLAGGED ISSUES : {val['flagged_issues']}")

        # Verification check
        if val["is_valid"] or val["score"] >= 80.0:
            print("STATUS         : [PASSED]")
            passed_count += 1
        else:
            print("STATUS         : [FAILED]")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_count} Tests Passed ({passed_count/total_count * 100:.1f}%)")
    print("=" * 80)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
