"""
================================================================================
CHEST AI: THREE-STAGE PIPELINE UNIT & REGRESSION TEST SUITE
================================================================================
Verifies:
  1. Automatic report passing from Stage 1 into Stage 2 (Diffusion Engine)
  2. Missing / empty reports are rejected and do not trigger reverse synthesis
  3. Anatomical safety gate halts pipeline for invalid / non-chest images
  4. Diffusion failure resilience (CUDA OOM, network/pipeline load failure)
  5. Prevention of placeholder / sample preset text in automatic workflow
  6. Integration of run_three_stage_pipeline in BidirectionalMedicalEngine
  7. Verification of synthetic labeling and research disclaimer
================================================================================
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from text_to_xray.config import (
    DiffusionConfig,
    PRIMARY_MODEL_ID,
    RESEARCH_SAFETY_DISCLAIMER
)
from text_to_xray.prompt_processor import (
    parse_chest_ai_report,
    adapt_report_to_diffusion_prompt
)
from text_to_xray.diffusion_engine import generate_synthetic_xray
from bidirectional_engine import BidirectionalMedicalEngine


class TestThreeStageAutomaticPassing(unittest.TestCase):
    """Tests automatic report passing from Stage 1 to Stage 2 diffusion."""

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_automatic_report_passing_to_diffusion(self, mock_load):
        """Verifies that an authentic generated report is passed into diffusion."""
        # Mock pipeline output
        fake_img = Image.fromarray(np.random.normal(120, 30, (512, 512, 3)).clip(0, 255).astype(np.uint8))
        mock_pipe = MagicMock()
        mock_out = MagicMock()
        mock_out.images = [fake_img]
        mock_pipe.return_value = mock_out
        mock_load.return_value = (mock_pipe, None)

        generated_doctor_report = (
            "FINDINGS: Moderate cardiomegaly is noted. Sternal wires are intact. "
            "No pleural effusion, pneumothorax, or pulmonary edema.\n\n"
            "IMPRESSION: 1. Moderate cardiomegaly. 2. No acute cardiopulmonary abnormality."
        )

        cfg = DiffusionConfig(model_id=PRIMARY_MODEL_ID, num_inference_steps=25)
        res = generate_synthetic_xray(
            report=generated_doctor_report,
            view="Frontal PA",
            seed=42,
            config=cfg,
            apply_watermark=True
        )

        self.assertTrue(res["success"])
        self.assertIsNotNone(res["image"])
        self.assertEqual(res["metadata"]["model_id"], PRIMARY_MODEL_ID)
        self.assertIn("cardiomegaly", res["metadata"]["conditioning"]["extracted_positive_findings"])
        self.assertIn("pleural effusion", res["metadata"]["conditioning"]["extracted_negative_findings"])

        # Clean up output artifacts
        if os.path.exists(res.get("image_path", "")):
            os.remove(res["image_path"])
        if os.path.exists(res.get("metadata_path", "")):
            os.remove(res["metadata_path"])

    def test_missing_report_does_not_trigger_diffusion(self):
        """Empty, None, or whitespace-only reports must be rejected immediately."""
        for invalid_rep in ["", None, "   \n\t  "]:
            res = generate_synthetic_xray(report=invalid_rep)
            self.assertFalse(res["success"])
            self.assertIn("Empty report", res["error"])
            self.assertIsNone(res["image"])

    def test_no_placeholder_presets_in_automatic_flow(self):
        """Ensures that default presets are not hardcoded or forced into the automatic flow."""
        from text_to_xray.prompt_processor import parse_chest_ai_report

        empty_report = ""
        parsed = parse_chest_ai_report(empty_report)
        self.assertEqual(parsed["positive_findings"], [])
        self.assertEqual(parsed["full_text"], "")


class TestAnatomicalSuitabilityHalt(unittest.TestCase):
    """Tests that anatomical suitability gate prevents report & diffusion generation for invalid images."""

    @patch("bidirectional_engine.validate_chest_xray")
    def test_invalid_image_halts_three_stage_pipeline(self, mock_validate):
        """Non-chest image (e.g. Knee or Abdomen) halts pipeline at Stage 1."""
        mock_validate.return_value = (
            False,
            0.15,
            "Knee",
            {
                "is_uncertain": False,
                "confidence_percentage": 15.0,
                "uncertainty_reason": "Classified as Knee"
            }
        )

        # Mock engine without loading huge weights
        with patch.object(BidirectionalMedicalEngine, "__init__", lambda self, *args, **kwargs: None):
            engine = BidirectionalMedicalEngine()
            # Provide dummy model & vocab attributes
            engine.model = MagicMock()
            engine.vocab = {}
            engine.word_to_index = {}
            engine.index_to_word = {}

            # Create a temporary dummy file
            dummy_path = os.path.join(PROJECT_DIR, "scratch", "test_knee_image.jpg")
            os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
            with open(dummy_path, "wb") as f:
                f.write(b"fake_image_content")

            try:
                result = engine.run_three_stage_pipeline(dummy_path)
                self.assertFalse(result["success"])
                self.assertEqual(result["stage_failed"], 1)
                self.assertTrue(result["stage_1_original"]["validation_failed"])
                self.assertEqual(result["stage_1_original"]["detected_label"], "Knee")
                self.assertIsNone(result["stage_2_report"])
                self.assertIsNone(result["stage_3_synthetic"])
            finally:
                if os.path.exists(dummy_path):
                    os.remove(dummy_path)


class TestDiffusionFailureResilience(unittest.TestCase):
    """Tests graceful handling when diffusion inference fails."""

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_diffusion_pipeline_loading_error(self, mock_load):
        """Pipeline loading error returns clean failure dict rather than uncaught exception."""
        mock_load.return_value = (None, "Repository P-RAJIV/cxr_stable_diffusion_v1_5 offline.")

        res = generate_synthetic_xray("FINDINGS: Normal chest radiograph.")
        self.assertFalse(res["success"])
        self.assertIn("Failed to initialize diffusion pipeline", res["error"])
        self.assertIsNone(res["image"])

    @patch("text_to_xray.diffusion_engine.load_diffusion_pipeline")
    def test_diffusion_oom_handling(self, mock_load):
        """CUDA Out of Memory in diffusion is caught and reported safely."""
        import torch
        mock_pipe = MagicMock()
        mock_pipe.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory during UNet denoising step.")
        mock_load.return_value = (mock_pipe, None)

        res = generate_synthetic_xray("FINDINGS: Mild bibasilar atelectasis.")
        self.assertFalse(res["success"])
        self.assertIn("CUDA Out of Memory", res["error"])
        self.assertIsNone(res["image"])


class TestBidirectionalEngineThreeStageIntegration(unittest.TestCase):
    """Tests the run_three_stage_pipeline method in BidirectionalMedicalEngine."""

    @patch.object(BidirectionalMedicalEngine, "__init__", lambda self, *args, **kwargs: None)
    @patch.object(BidirectionalMedicalEngine, "forward_predict")
    @patch.object(BidirectionalMedicalEngine, "refine_report")
    @patch("bidirectional_engine.generate_synthetic_xray")
    def test_complete_three_stage_pipeline_success(self, mock_gen, mock_refine, mock_forward):
        engine = BidirectionalMedicalEngine()

        # Mock forward path (Stage 1)
        mock_forward.return_value = {
            "validation_failed": False,
            "pathology_predictions": {"Cardiomegaly": 0.85, "Effusion": 0.12},
            "raw_report": "cardiomegaly is present"
        }

        # Mock refinement (Stage 1.5)
        mock_refine.return_value = {
            "medical_report": {
                "findings": "Cardiomegaly is present. Clear lungs without effusion.",
                "impression": "Moderate cardiomegaly."
            },
            "findings": "Cardiomegaly is present. Clear lungs without effusion.",
            "impression": "Moderate cardiomegaly.",
            "simple_report": "Your heart appears slightly larger than normal.",
            "validation": {"is_valid": True, "score": 100.0}
        }

        # Mock reverse diffusion (Stage 2)
        fake_synth_img = Image.fromarray(np.random.normal(120, 30, (512, 512, 3)).clip(0, 255).astype(np.uint8))
        mock_gen.return_value = {
            "success": True,
            "image": fake_synth_img,
            "image_path": "synthetic_outputs/images/test_cxr.png",
            "metadata_path": "synthetic_outputs/metadata/test_cxr.json",
            "metadata": {
                "model_id": PRIMARY_MODEL_ID,
                "generation_parameters": {"seed": 42}
            }
        }

        dummy_img = os.path.join(PROJECT_DIR, "scratch", "dummy_valid_cxr.jpg")
        os.makedirs(os.path.dirname(dummy_img), exist_ok=True)
        with open(dummy_img, "wb") as f:
            f.write(b"dummy")

        try:
            res = engine.run_three_stage_pipeline(dummy_img, seed=42)
            self.assertTrue(res["success"])

            # Verify Stage 1 results
            self.assertIn("stage_1_original", res)
            self.assertTrue(res["stage_1_original"]["validation_passed"])
            self.assertEqual(res["stage_1_original"]["pathology_predictions"]["Cardiomegaly"], 0.85)

            # Verify Stage 2 report results
            self.assertIn("stage_2_report", res)
            self.assertIn("Moderate cardiomegaly", res["stage_2_report"]["doctor_report"])
            self.assertEqual(res["stage_2_report"]["simple_report"], "Your heart appears slightly larger than normal.")

            # Verify Stage 3 synthetic results
            self.assertIn("stage_3_synthetic", res)
            self.assertTrue(res["stage_3_synthetic"]["success"])
            self.assertIsNotNone(res["stage_3_synthetic"]["image"])

            # Verify Safety Disclaimer is attached
            self.assertEqual(res["safety_disclaimer"], RESEARCH_SAFETY_DISCLAIMER)

            # Verify report passed to generate_synthetic_xray
            mock_gen.assert_called_once()
            called_kwargs = mock_gen.call_args[1]
            self.assertIn("Cardiomegaly", called_kwargs["report"])
        finally:
            if os.path.exists(dummy_img):
                os.remove(dummy_img)


if __name__ == "__main__":
    unittest.main(verbosity=2)
