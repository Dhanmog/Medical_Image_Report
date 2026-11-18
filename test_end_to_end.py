"""
================================================================================
END-TO-END INTEGRATION TEST FOR CHEST X-RAY DUAL REPORT SYSTEM
================================================================================
Simulates the complete pipeline:
  Image / Raw Text -> Model Prediction -> Dual Report Refinement -> Quality Control
================================================================================
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from report_refinement import refine_report, validate_report_consistency

def test_full_pipeline():
    print("=" * 80)
    print("  RUNNING END-TO-END PIPELINE VALIDATION TEST")
    print("=" * 80)

    sample_predictions = [
        "The cardiac silhouette is enlarged with a left-sided pleural effusion.",
        "Right lower lung infection with patchy cloudy area suggestive of pneumonia.",
        "The heart is big and there is fluid around the left lung.",
        "The cardiac silhouette and lung fields are within normal limits.",
        "Air around the right lung present."
    ]

    all_passed = True

    for idx, raw in enumerate(sample_predictions, 1):
        print(f"\n[SAMPLE {idx}] Raw Input: '{raw}'")
        result = refine_report(raw)
        
        assert "simple_report" in result, "Missing simple_report key"
        assert "medical_report" in result, "Missing medical_report key"
        assert "findings" in result, "Missing findings key"
        assert "impression" in result, "Missing impression key"
        assert "validation" in result, "Missing validation key"

        print("  -> Simple Report       : ", result["simple_report"])
        print("  -> Professional Findings: ", result["findings"])
        print("  -> Impression          : ", result["impression"])
        print(f"  -> Validation Status   : Score={result['validation']['score']}%, Valid={result['validation']['is_valid']}")

        if result["validation"]["score"] < 80.0:
            print("  ❌ Test Failed!")
            all_passed = False
        else:
            print("  ✅ Test Passed!")

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL END-TO-END PIPELINE TESTS PASSED SUCCESSFULLY!")
    else:
        print("  SOME TESTS FAILED - PLEASE REVIEW")
    print("=" * 80)
    return all_passed

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
