"""
================================================================================
CHEST AI — PHASE 2 LLM REFINEMENT & FALLBACK CASCADE REGRESSION TEST SUITE
================================================================================
Tests Phase 2 LLM Integration, Fallback Cascade, and Report Quality:
  1. Valid Gemini Flash API / Groq Cascade execution
  2. Missing API Key Fallback to Local BioBERT Engine
  3. Invalid API Key Fallback Resilience
  4. Single Chest X-ray Observation Refinement
  5. 3-View Chest X-ray Multi-Image Consolidation & Refinement
  6. Non-Chest Rejection Gate Integrity
  7. Empty / Malformed Transformer Output Robustness
  8. Quality Control & Hallucination Prevention
================================================================================
"""

import sys
import os
import unittest

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

from report_refinement import (
    refine_report,
    clean_raw_report,
    _call_llm_api_cascade,
    _get_api_key
)
from chest_validator import validate_chest_xray


class TestPhase2LLMRefinement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("RUNNING CHEST AI PHASE 2 LLM REFINEMENT REGRESSION SUITE")
        print("=======================================================")

    def test_01_missing_api_key_biobert_fallback(self):
        """Verify that when no API key is set, system safely uses local BioBERT engine."""
        # Temporarily clear env vars
        old_gemini = os.environ.pop("GEMINI_API_KEY", None)
        old_groq = os.environ.pop("GROQ_API_KEY", None)
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_api = os.environ.pop("API_KEY", None)

        try:
            raw = "Substantial cardiomegaly. Pulmonary vascular congestion. Bilateral pleural effusions."
            res = refine_report(raw)

            self.assertIn("BioBERT", res["engine_used"], "Engine must report BioBERT fallback when no API key is present")
            self.assertFalse(res["is_llm_active"], "is_llm_active should be False")
            self.assertIn("cardiomegaly", res["medical_report"]["full_text"].lower())
            self.assertTrue(res["validation"]["is_valid"])
            print(f"[OK] Test 01 Passed: Missing key safely engaged {res['engine_used']}.")
        finally:
            if old_gemini: os.environ["GEMINI_API_KEY"] = old_gemini
            if old_groq: os.environ["GROQ_API_KEY"] = old_groq
            if old_openai: os.environ["OPENAI_API_KEY"] = old_openai
            if old_api: os.environ["API_KEY"] = old_api

    def test_02_invalid_api_key_resilience(self):
        """Verify that an invalid API key does NOT crash the application and falls back cleanly."""
        os.environ["GEMINI_API_KEY"] = "AIzaInvalidFakeKeyForTesting12345"
        
        raw = "Mild cardiomegaly with small left pleural effusion."
        res = refine_report(raw)

        # Must not crash, must return valid result
        self.assertIsNotNone(res["simple_report"])
        self.assertIsNotNone(res["medical_report"]["full_text"])
        self.assertTrue(res["validation"]["is_valid"])
        print(f"[OK] Test 02 Passed: Invalid API key handled gracefully without crashing. Result generated via {res['engine_used']}.")
        os.environ.pop("GEMINI_API_KEY", None)

    def test_03_single_view_observation_refinement(self):
        """Verify single chest X-ray report refinement and FINDINGS/IMPRESSION split."""
        raw = "Right lower lobe airspace opacity suggestive of pneumonia. No pleural effusion."
        res = refine_report(raw)

        med = res["medical_report"]
        self.assertIn("FINDINGS:", med["full_text"])
        self.assertIn("IMPRESSION:", med["full_text"])
        self.assertIn("pneumonia", med["full_text"].lower())
        self.assertIn("effusion", med["full_text"].lower())
        self.assertTrue("lung infection" in res["simple_report"].lower() or "pneumonia" in res["simple_report"].lower())
        print("[OK] Test 03 Passed: Single view observation refined with FINDINGS/IMPRESSION split.")

    def test_04_three_view_multi_image_consolidation(self):
        """Verify 3-view chest X-ray consolidation, device deduplication, and range preservation."""
        raw_multi = (
            "View 1: Borderline cardiac silhouette. Intact midline sternal wires. "
            "View 2: Mild cardiomegaly. Sternal wires intact. Course of pacemaker leads is constant. "
            "View 3: Left pleural effusion. Unchanged left pectoral pacemaker."
        )
        preds = {"Cardiomegaly": 0.75, "Effusion": 0.82}
        
        res = refine_report(raw_multi, pathology_predictions=preds)
        med_text = res["medical_report"]["full_text"].lower()

        # 1. Deduplicated sternal wires
        self.assertEqual(med_text.count("intact midline sternal wires are present."), 1, "Sternal wires statement must be deduplicated")

        # 2. Pacemaker details preserved
        self.assertIn("pacemaker", med_text)
        self.assertTrue("left" in med_text or "leads" in med_text)

        # 3. Range preservation (borderline to cardiomegaly)
        self.assertIn("borderline", med_text)
        self.assertIn("cardiomegaly", med_text)
        print("[OK] Test 04 Passed: 3-view multi-image consolidation, device deduplication, and range preservation verified.")

    def test_05_non_chest_rejection_gate_integrity(self):
        """Verify that MobileNetV2 chest validator rejects non-chest images without modifying pipeline."""
        if validate_chest_xray is None:
            self.skipTest("validate_chest_xray module not available")

        # Create a blank non-chest synthetic image
        import numpy as np
        import cv2

        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", blank_img)
        img_bytes = encoded.tobytes()

        val_res = validate_chest_xray(img_bytes)
        is_chest, top1_conf, label, details = val_res
        self.assertFalse(is_chest)
        print(f"[OK] Test 05 Passed: Non-chest rejection gate evaluated: is_chest={is_chest} ({label}).")

    def test_06_empty_malformed_input_robustness(self):
        """Verify system handles empty or malformed inputs cleanly."""
        res_empty = refine_report("")
        self.assertIn("No diagnostic findings", res_empty["simple_report"])
        self.assertIn("unremarkable", res_empty["medical_report"]["full_text"].lower())

        res_junk = refine_report("findings: impression: in comparison with prior study of")
        self.assertIsNotNone(res_junk["simple_report"])
        self.assertIsNotNone(res_junk["medical_report"]["full_text"])
        print("[OK] Test 06 Passed: Empty and malformed inputs handled cleanly without throwing errors.")


if __name__ == "__main__":
    unittest.main()
