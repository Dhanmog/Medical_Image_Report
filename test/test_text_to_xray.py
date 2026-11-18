"""
================================================================================
CHEST AI: UNIT & INTEGRATION REGRESSION TEST SUITE FOR TEXT-TO-XRAY MODULE
================================================================================
Verifies:
  1. Report parsing and finding extraction
  2. Strict negation preservation (zero inversion)
  3. Laterality and anatomical location preservation
  4. Clinical uncertainty preservation
  5. Projection handling (PA, AP, Lateral, Unknown)
  6. Empty / missing report handling
  7. Model loader error handling
  8. CUDA Out-Of-Memory (OOM) handling
  9. Generation failure resilience
  10. Technical image validation (rejects blank, black, degenerate images)
  11. Synthetic research watermark labeling
  12. Audit metadata integrity
  13. Compatibility with BidirectionalMedicalEngine
  14. Anatomical Safety Gate regression
  15. Dual-report refinement regression
================================================================================
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# Path setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from text_to_xray.config import DiffusionConfig, RESEARCH_SAFETY_DISCLAIMER
from text_to_xray.prompt_processor import (
    parse_chest_ai_report,
    adapt_report_to_diffusion_prompt,
    is_term_negated,
    extract_laterality,
    extract_uncertainty
)
from text_to_xray.image_validator import (
    validate_synthetic_image,
    apply_synthetic_watermark
)
from text_to_xray.diffusion_engine import generate_synthetic_xray
from text_to_xray.model_loader import clear_pipeline_cache


class TestPromptProcessor(unittest.TestCase):
    """Tests clinical text processing, negation, laterality, and diffusion prompt adaptation."""

    def test_correct_report_parsing(self):
        report = (
            "FINDINGS: Cardiomegaly is present. Sternal wires are intact. "
            "No pleural effusion or pneumothorax.\n"
            "IMPRESSION: 1. Moderate cardiomegaly. 2. No acute pulmonary disease."
        )
        parsed = parse_chest_ai_report(report)
        self.assertIn("cardiomegaly", parsed["positive_findings"])
        self.assertIn("pleural effusion", parsed["negative_findings"])
        self.assertIn("pneumothorax", parsed["negative_findings"])

    def test_negation_preservation_zero_inversion(self):
        """CRITICAL: 'No pleural effusion' must NEVER become a positive finding."""
        report = "Lungs are well expanded. No pleural effusion. No pneumothorax. No focal pneumonia."
        parsed = parse_chest_ai_report(report)

        self.assertNotIn("pleural effusion", parsed["positive_findings"])
        self.assertNotIn("pneumothorax", parsed["positive_findings"])
        self.assertNotIn("pneumonia", parsed["positive_findings"])

        self.assertIn("pleural effusion", parsed["negative_findings"])
        self.assertIn("pneumothorax", parsed["negative_findings"])
        self.assertIn("pneumonia", parsed["negative_findings"])

        bundle = adapt_report_to_diffusion_prompt(report)
        self.assertIn("pleural effusion", bundle["negative_prompt"])
        self.assertNotIn("showing pleural effusion", bundle["positive_prompt"])

    def test_laterality_preservation(self):
        """Preserves left-sided vs right-sided laterality."""
        report = "There is a small left pleural effusion. The right lung base is clear."
        parsed = parse_chest_ai_report(report)
        self.assertIn("pleural effusion", parsed["positive_findings"])
        self.assertEqual(parsed["laterality"].get("pleural effusion"), "left-sided")

        bundle = adapt_report_to_diffusion_prompt(report)
        self.assertIn("left-sided pleural effusion", bundle["positive_prompt"])

    def test_uncertainty_preservation(self):
        """Verifies that borderline or possible diagnoses remain flagged as uncertain."""
        report = "Patchy opacity in right lower lobe, possible pneumonia. Cannot exclude aspiration."
        parsed = parse_chest_ai_report(report)
        self.assertIn("pneumonia", parsed["positive_findings"])
        self.assertTrue(parsed["uncertainty"].get("pneumonia"))

        bundle = adapt_report_to_diffusion_prompt(report)
        self.assertIn("possible", bundle["positive_prompt"])

    def test_unknown_and_custom_projection_handling(self):
        """Handles reports without projection and permits user override."""
        report = "Lungs are clear. Normal cardiac silhouette."
        parsed = parse_chest_ai_report(report)
        self.assertEqual(parsed["projection"], "Unknown")

        # Test user override
        bundle_pa = adapt_report_to_diffusion_prompt(report, user_projection="Frontal PA")
        self.assertEqual(bundle_pa["projection_used"], "Frontal PA")
        self.assertIn("Frontal PA chest radiograph", bundle_pa["positive_prompt"])

    def test_missing_report_handling(self):
        """Empty report strings or empty dicts return structured failure without raising unhandled exceptions."""
        res_empty = generate_synthetic_xray("")
        self.assertFalse(res_empty["success"])
        self.assertIn("Empty report", res_empty["error"])

        res_none = generate_synthetic_xray(None)
        self.assertFalse(res_none["success"])


class TestImageValidatorAndLabeling(unittest.TestCase):
    """Tests technical image validation, degenerate detection, and watermark labeling."""

    def test_reject_none_or_invalid_type(self):
        is_val, err, _ = validate_synthetic_image(None)
        self.assertFalse(is_val)

        is_val, err, _ = validate_synthetic_image("not_an_image")
        self.assertFalse(is_val)

    def test_reject_degenerate_black_image(self):
        black_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        is_val, err, metrics = validate_synthetic_image(black_img)
        self.assertFalse(is_val)
        self.assertTrue(metrics["is_degenerate"])

    def test_reject_degenerate_white_image(self):
        white_img = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))
        is_val, err, metrics = validate_synthetic_image(white_img)
        self.assertFalse(is_val)
        self.assertTrue(metrics["is_degenerate"])

    def test_accept_valid_radiograph_like_image(self):
        # Generate an image with realistic variance
        np.random.seed(42)
        valid_arr = np.random.normal(loc=120, scale=35, size=(512, 512, 3)).clip(0, 255).astype(np.uint8)
        valid_img = Image.fromarray(valid_arr)

        is_val, err, metrics = validate_synthetic_image(valid_img)
        self.assertTrue(is_val)
        self.assertIsNone(err)
        self.assertFalse(metrics["is_degenerate"])

    def test_synthetic_output_labeling(self):
        """Verifies watermark banner application for research transparency."""
        base_img = Image.fromarray(np.full((512, 512, 3), 100, dtype=np.uint8))
        watermarked = apply_synthetic_watermark(base_img)

        self.assertIsInstance(watermarked, Image.Image)
        self.assertEqual(watermarked.size, (512, 512))
        
        # Check that top rows contain the dark banner
        banner_px = np.array(watermarked)[:10, :10]
        self.assertLess(np.mean(banner_px), 50)  # Dark banner bar


class TestDiffusionEngineMocked(unittest.TestCase):
    """Tests diffusion engine execution, OOM resilience, and audit metadata using mocks."""

    def setUp(self):
        clear_pipeline_cache()

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_generation_success_and_metadata_audit(self, mock_load):
        # Mock valid diffusion output
        np.random.seed(123)
        fake_cxr = Image.fromarray(np.random.normal(110, 40, (512, 512, 3)).clip(0, 255).astype(np.uint8))
        
        mock_pipeline = MagicMock()
        mock_output = MagicMock()
        mock_output.images = [fake_cxr]
        mock_pipeline.return_value = mock_output
        mock_load.return_value = (mock_pipeline, None)

        report = "FINDINGS: Cardiomegaly and bilateral pleural effusions.\nIMPRESSION: Congestive failure."
        res = generate_synthetic_xray(report, view="Frontal PA", seed=99)

        self.assertTrue(res["success"])
        self.assertIsNotNone(res["image"])
        self.assertTrue(os.path.exists(res["image_path"]))
        self.assertTrue(os.path.exists(res["metadata_path"]))

        # Verify metadata audit content
        meta = res["metadata"]
        self.assertEqual(meta["generation_parameters"]["seed"], 99)
        self.assertIn("cardiomegaly", meta["conditioning"]["extracted_positive_findings"])
        self.assertIn("pleural effusion", meta["conditioning"]["extracted_positive_findings"])
        self.assertEqual(meta["safety_disclaimer"], RESEARCH_SAFETY_DISCLAIMER)

        # Cleanup created files
        if os.path.exists(res["image_path"]):
            os.remove(res["image_path"])
        if os.path.exists(res["metadata_path"]):
            os.remove(res["metadata_path"])

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_model_loading_failure_handled(self, mock_load):
        mock_load.return_value = (None, "Repository not reachable or gated.")

        res = generate_synthetic_xray("FINDINGS: Normal lungs.")
        self.assertFalse(res["success"])
        self.assertIn("Failed to initialize diffusion pipeline", res["error"])

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_insufficient_gpu_memory_handled(self, mock_load):
        import torch
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory in UNet forward pass.")
        mock_load.return_value = (mock_pipeline, None)

        res = generate_synthetic_xray("FINDINGS: Left pneumonia.")
        self.assertFalse(res["success"])
        self.assertIn("CUDA Out of Memory", res["error"])


class TestPipelineRegression(unittest.TestCase):
    """Ensures existing forward diagnostic models, safety gate, and refinement remain intact."""

    def test_anatomical_safety_gate_regression(self):
        """Verifies chest_validator module imports and thresholding remain functional."""
        from chest_validator import DEFAULT_CHEST_THRESHOLD, validate_chest_xray
        self.assertIsNotNone(validate_chest_xray)
        self.assertAlmostEqual(DEFAULT_CHEST_THRESHOLD, 0.60, places=2)

    def test_report_refinement_regression(self):
        """Verifies report_refinement operates without regression."""
        from report_refinement import refine_report
        sample_raw = "cardiomegaly is present without pleural effusion"
        result = refine_report(sample_raw)
        self.assertIn("findings", result)
        self.assertIn("impression", result)
        self.assertIn("simple_report", result)

    @patch("bidirectional_engine.generate_synthetic_xray")
    def test_bidirectional_engine_backward_path_integration(self, mock_gen):
        """Verifies BidirectionalMedicalEngine backward path utilizes diffusion when provided."""
        from bidirectional_engine import BidirectionalMedicalEngine

        # Mock successful generation
        mock_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        mock_gen.return_value = {
            "success": True,
            "image": mock_img,
            "image_path": os.path.join(PROJECT_DIR, "scratch", "test_mock_recon.png"),
            "metadata": {"test": True}
        }

        test_img_path = os.path.join(PROJECT_DIR, "xray1.jpg")
        if os.path.exists(test_img_path):
            engine = BidirectionalMedicalEngine()
            bkwd = engine.backward_reconstruct_and_localize(
                image_path=test_img_path,
                top_pathology="Cardiomegaly",
                report="FINDINGS: Normal chest.\nIMPRESSION: Normal.",
                use_diffusion=True
            )
            self.assertIn("reconstructed_image", bkwd)
            self.assertIn("reconstructed_image_path", bkwd)
            self.assertTrue(bkwd["is_diffusion_generated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
