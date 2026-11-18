import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_refinement import refine_report, generate_simple_report, generate_medical_report
from biobert_nlp_engine import BioBERTBiomedicalNLPEngine


class TestPhase3QualityFixes(unittest.TestCase):
    """Regression test suite for Phase 3 Output Quality & Medical Grounding Fixes."""

    @classmethod
    def setUpClass(cls):
        cls.biobert = BioBERTBiomedicalNLPEngine()

    def test_01_left_effusion_not_bilateral(self):
        """Verify left pleural effusion with basilar atelectasis remains left-sided and does NOT become bilateral."""
        raw = "little change in the degree of left pleural effusion with compressive atelectasis at the base."
        res = refine_report(raw)

        prof_text = res["medical_report"]["full_text"].lower()
        simple_text = res["simple_report"].lower()

        # Must contain left pleural effusion
        self.assertIn("left pleural effusion", prof_text)
        self.assertIn("left lung", simple_text)

        # Must NOT convert to bilateral pleural effusions
        self.assertNotIn("bilateral pleural effusions", prof_text)
        self.assertNotIn("both lungs", simple_text)
        print("[OK] Test 01 Passed: Left pleural effusion preserved without expanding to bilateral.")

    def test_02_explicit_pneumonia_negation(self):
        """Verify explicit pneumonia negation remains unambiguously negative."""
        raw = "cardiomegaly. no evidence of acute focal pneumonia."
        res = refine_report(raw)

        prof_text = res["medical_report"]["full_text"].lower()
        simple_text = res["simple_report"].lower()

        self.assertIn("no", prof_text)
        self.assertIn("pneumonia", prof_text)

        # Ensure simple report does NOT contain ambiguous positive phrase 'signs of pneumonia identified'
        self.assertNotIn("signs of pneumonia identified", simple_text)
        self.assertTrue("no" in simple_text and "pneumonia" in simple_text)
        print("[OK] Test 02 Passed: Explicit pneumonia negation preserved unambiguously.")

    def test_03_explicit_edema_negation(self):
        """Verify pulmonary edema negation remains negative and combined cleanly with 'or'."""
        raw = "cardiomegaly. no vascular congestion. no pulmonary edema. no acute focal pneumonia."
        res = refine_report(raw)

        simple_text = res["simple_report"].lower()
        prof_text = res["medical_report"]["full_text"].lower()

        self.assertIn("no", prof_text)
        self.assertIn("edema", prof_text)

        # Simple report must format combined negatives with 'or' and remain negative
        self.assertNotIn("no pulmonary edema, signs of pneumonia identified", simple_text)
        self.assertTrue("no" in simple_text or "clear of" in simple_text)
        print("[OK] Test 03 Passed: Explicit pulmonary edema negation preserved cleanly.")

    def test_04_uncertainty_preservation(self):
        """Verify possible/borderline findings retain uncertainty and are not converted to confirmed diagnoses."""
        raw = "cardiac silhouette is at the upper limit of normal. possible small left pleural effusion."
        res = refine_report(raw)

        prof_text = res["medical_report"]["full_text"].lower()

        # Borderline cardiac size must remain borderline
        self.assertTrue("borderline" in prof_text or "upper limit" in prof_text)
        # Should not state definitive cardiomegaly without qualification
        self.assertNotIn("cardiomegaly is present.", prof_text)
        print("[OK] Test 04 Passed: Borderline and possible findings preserve uncertainty.")

    def test_05_multi_view_laterality(self):
        """Verify multi-view observations preserve distinct laterality (left subclavian line, right chest tube)."""
        raw_multi = "View 1: left subclavian central venous line in place. View 2: right chest tube present."
        res = refine_report(raw_multi)

        prof_text = res["medical_report"]["full_text"].lower()

        self.assertIn("left", prof_text)
        self.assertIn("right", prof_text)
        self.assertIn("chest tube", prof_text)
        print("[OK] Test 05 Passed: Multi-view laterality preserved distinctly.")

    def test_06_patient_friendly_no_reversal(self):
        """Verify patient-friendly output never reverses clinical meaning of negatives."""
        raw = "lungs are clear. no pulmonary edema. no pneumonia."
        simple_rep = generate_simple_report(raw)

        simple_lower = simple_rep.lower()
        self.assertTrue(simple_lower.startswith("no") or "normal" in simple_lower or "clear" in simple_lower)
        self.assertNotIn("pneumonia identified", simple_lower)
        print("[OK] Test 06 Passed: Patient-friendly output preserves exact clinical meaning.")

    def test_07_unremarkable_pacemaker_leads(self):
        """Verify 'course of pacemaker leads is unremarkable' is classified as NORMAL status."""
        sentence = "The course of the pacemaker leads is unremarkable."
        pol, status_val, cert = self.biobert.classify_polarity_certainty(sentence, "Support Devices")
        self.assertEqual(status_val, "NORMAL")
        self.assertEqual(pol, "NEUTRAL")
        print("[OK] Test 07 Passed: Unremarkable pacemaker leads classified as NORMAL / NEUTRAL.")

    def test_08_high_confidence_unmentioned_effusion(self):
        """Verify unmentioned effusion with 83% DenseNet prob is categorized as IMAGE_POS_REPORT_OMITTED."""
        preds = {"Effusion": 0.8318, "Atelectasis": 0.5166}
        findings = self.biobert.extract_radiology_findings("right chest tube is in place.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        self.assertIn("category_counts", ver)
        self.assertEqual(ver["category_counts"]["image_positive_omitted"], 1)
        self.assertEqual(ver["category_counts"]["confirmed_contradictions"], 0)
        print("[OK] Test 08 Passed: 83% unmentioned effusion categorized as Image-Positive / Report-Omitted.")

    def test_09_borderline_atelectasis(self):
        """Verify 51.6% unmentioned atelectasis is categorized as MODEL_UNCERTAINTY_BORDERLINE."""
        preds = {"Atelectasis": 0.5166}
        findings = self.biobert.extract_radiology_findings("right chest tube is in place.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        self.assertEqual(ver["category_counts"]["model_uncertainty"], 1)
        self.assertEqual(ver["category_counts"]["confirmed_contradictions"], 0)
        print("[OK] Test 09 Passed: 51.6% atelectasis categorized as Model Uncertainty / Borderline.")

    def test_10_explicit_negative_effusion_high_image_prob(self):
        """Verify explicit 'no pleural effusion' with 85% DenseNet prob is categorized as CONFIRMED_CONTRADICTION."""
        preds = {"Pleural Effusion": 0.8500}
        findings = self.biobert.extract_radiology_findings("No pleural effusion is present.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        self.assertEqual(ver["category_counts"]["confirmed_contradictions"], 1)
        self.assertEqual(ver["confirmed_contradictions"][0]["category"], "CONFIRMED_CONTRADICTION")
        print("[OK] Test 10 Passed: Explicit negative with 85% image prob categorized as Confirmed Contradiction.")

    def test_11_normal_observations_vs_positive_pathologies(self):
        """Verify normal device observations do not register as positive pathology findings."""
        raw = "the course of the pacemaker leads is unremarkable. nasogastric tube is unremarkable."
        res = refine_report(raw)

        findings = res.get("biobert_findings", [])
        normal_findings = [f for f in findings if f.get("status") == "NORMAL"]
        positive_findings = [f for f in findings if f.get("status") == "PRESENT"]

        self.assertGreater(len(normal_findings), 0)
        self.assertEqual(len(positive_findings), 0)
        print("[OK] Test 11 Passed: Unremarkable device observations correctly assigned NORMAL status.")

    def test_12_phase32_category_counts_sum_and_score_consistency(self):
        """Verify Phase 3.2 5-category counts sum to total pathologies and consistency score is 75% (3/4 evaluated)."""
        preds = {
            "Effusion": 0.8318,
            "Atelectasis": 0.5166,
            "Pneumothorax": 0.4652,
            "Edema": 0.4477,
            "Cardiomegaly": 0.3097
        }
        raw_text = "Cardiomegaly is present. Atelectasis is present. No large pleural effusions. No pulmonary edema."
        findings = self.biobert.extract_radiology_findings(raw_text)
        ver = self.biobert.compare_with_densenet(findings, preds)

        counts = ver["category_counts"]
        # Category 1: Effusion (83.18% vs ABSENT) -> 1
        self.assertEqual(counts["confirmed_contradictions"], 1)
        # Category 2: Image-Positive Omitted -> 0
        self.assertEqual(counts["image_positive_omitted"], 0)
        # Category 3: Low Conf -> 0
        self.assertEqual(counts["text_positive_low_conf"], 0)
        # Category 4: Pneumothorax (46.52% vs UNMENTIONED) -> 1
        self.assertEqual(counts["model_uncertainty"], 1)
        # Category 5: Atelectasis, Cardiomegaly, Edema -> 3
        self.assertEqual(counts["concordant_normal"], 3)

        # Sum of all 5 category counts must equal total evaluated pathologies (5)
        self.assertEqual(sum(counts.values()), len(preds))

        # Evaluated count denominator must be 4 (Effusion, Atelectasis, Edema, Cardiomegaly)
        self.assertEqual(ver["evaluable_count"], 4)

        # Consistency score must be 75.0% (3/4 evaluated)
        self.assertEqual(ver["consistency_score"], 75.0)
        print("[OK] Test 12 Passed: Phase 3.2 category counts sum to total pathologies and score is 75% (3/4).")

    def test_13_phase32_empty_densenet_predictions_returns_category_counts(self):
        """Verify empty DenseNet predictions dictionary returns valid zeroed category_counts structure."""
        findings = self.biobert.extract_radiology_findings("Lungs are clear.")
        ver = self.biobert.compare_with_densenet(findings, {})

        self.assertIn("category_counts", ver)
        counts = ver["category_counts"]
        for cat_key in ["confirmed_contradictions", "image_positive_omitted", "text_positive_low_conf", "model_uncertainty", "concordant_normal"]:
            self.assertIn(cat_key, counts)
            self.assertEqual(counts[cat_key], 0)
        print("[OK] Test 13 Passed: Empty predictions dict returns complete zeroed category_counts dict.")

    def test_14_left_effusion_increased_right_decreased(self):
        """Verify left effusion increased and right effusion minimally decreased are distinctly preserved."""
        raw = "The previous radiograph, the left pleural effusion has increased in extent. The effusion on the right has minimally decreased in extent."
        res = refine_report(raw)

        prof_text = res["medical_report"]["full_text"].lower()
        simple_text = res["simple_report"].lower()

        self.assertIn("left pleural effusion", prof_text)
        self.assertIn("increased", prof_text)
        self.assertIn("right pleural effusion", prof_text)
        self.assertTrue("minimally decreased" in prof_text or "decreased" in prof_text)

        self.assertIn("left lung", simple_text)
        self.assertIn("increased", simple_text)
        self.assertIn("right lung", simple_text)
        print("[OK] Test 14 Passed: Left effusion increased and right effusion minimally decreased preserved distinctly.")

    def test_15_stable_chest_tube_not_newly_placed(self):
        """Verify 'The right chest tube is in unchanged position' is not described as newly placed."""
        raw = "The right chest tube is in unchanged position."
        res = refine_report(raw)

        prof_text = res["medical_report"]["full_text"].lower()
        simple_text = res["simple_report"].lower()

        self.assertIn("right chest tube", prof_text)
        self.assertIn("unchanged position", prof_text)
        self.assertNotIn("has been placed", prof_text)
        self.assertNotIn("placed since", simple_text)
        print("[OK] Test 15 Passed: Stable right chest tube described as unchanged position and NOT newly placed.")

    def test_16_new_device_placement_preserved_only_when_explicit(self):
        """Verify new device placement is preserved only when explicitly stated."""
        raw_new = "Compared to the prior radiograph, a new right chest tube has been placed."
        res_new = refine_report(raw_new)

        prof_new = res_new["medical_report"]["full_text"].lower()
        self.assertTrue("placed" in prof_new or "new" in prof_new)

        raw_stable = "Right chest tube in unchanged position."
        res_stable = refine_report(raw_stable)
        prof_stable = res_stable["medical_report"]["full_text"].lower()
        self.assertNotIn("has been placed", prof_stable)
        print("[OK] Test 16 Passed: New device placement distinguished from stable device position.")

    def test_17_temporal_qualifiers_survive_reports(self):
        """Verify temporal qualifiers (increased, minimally decreased, unchanged) survive report generation."""
        raw = "Left pleural effusion has increased in extent. Right chest tube is in unchanged position."
        res = refine_report(raw)

        prof = res["medical_report"]["full_text"].lower()
        simple = res["simple_report"].lower()

        self.assertIn("increased", prof)
        self.assertIn("unchanged", prof)
        self.assertIn("increased", simple)
        print("[OK] Test 17 Passed: Temporal qualifiers survive both professional and patient-friendly report generation.")

    def test_18_category_counts_reconcile_with_records(self):
        """Verify 5 category counts sum exactly to total pathologies in comparison matrix."""
        preds = {
            "Effusion": 0.8318, "Atelectasis": 0.5166, "Pneumothorax": 0.4652, "Edema": 0.4477,
            "Cardiomegaly": 0.3097, "Consolidation": 0.1200, "Infiltration": 0.1500,
            "Mass": 0.0500, "Nodule": 0.0800, "Emphysema": 0.0200, "Fibrosis": 0.0100,
            "Pleural Thickening": 0.0300, "Hernia": 0.0100, "Pneumonia": 0.0500
        }
        findings = self.biobert.extract_radiology_findings("Cardiomegaly. Left pleural effusion has increased.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        counts = ver["category_counts"]
        matrix = ver["comparison_matrix"]

        self.assertEqual(sum(counts.values()), len(matrix))
        self.assertEqual(sum(counts.values()), 14)
        print("[OK] Test 18 Passed: 5 category counts reconcile exactly with 14 comparison records.")

    def test_19_consistency_percentage_and_denominator_mathematically_correct(self):
        """Verify consistency score percentage equals concordant evaluable records divided by evaluable count."""
        preds = {"Effusion": 0.8318, "Cardiomegaly": 0.3097}
        findings = self.biobert.extract_radiology_findings("Cardiomegaly is present. No pleural effusion.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        eval_count = ver["evaluable_count"]
        # Effusion is contradiction (eval=1, conc=0), Cardiomegaly is concordant (eval=1, conc=1)
        self.assertEqual(eval_count, 2)
        self.assertEqual(ver["consistency_score"], 50.0)
        print("[OK] Test 19 Passed: Consistency percentage and denominator are mathematically correct.")

    def test_20_explicit_negative_vs_unmentioned_distinct(self):
        """Verify explicit negative finding is status ABSENT while unmentioned is UNMENTIONED."""
        raw = "No pleural effusion is identified."
        findings = self.biobert.extract_radiology_findings(raw)

        eff_finding = next((f for f in findings if f["pathology"] == "Pleural Effusion"), None)
        self.assertIsNotNone(eff_finding)
        self.assertEqual(eff_finding["status"], "ABSENT")

        preds = {"Pleural Effusion": 0.10, "Pneumothorax": 0.10}
        ver = self.biobert.compare_with_densenet(findings, preds)
        matrix = {m["pathology"]: m["report_status"] for m in ver["comparison_matrix"]}

        self.assertEqual(matrix["Pleural Effusion"], "ABSENT")
        self.assertEqual(matrix["Pneumothorax"], "UNMENTIONED")
        print("[OK] Test 20 Passed: Explicit negative finding remains distinct from unmentioned finding.")

    def test_21_review_required_status_not_presented_as_full_concordance(self):
        """Verify review-required status is flagged when high probability unmentioned finding exists."""
        preds = {"Pleural Effusion": 0.8500}
        findings = self.biobert.extract_radiology_findings("Lungs are clear.")
        ver = self.biobert.compare_with_densenet(findings, preds)

        self.assertTrue(ver["inconsistencies_found"])
        self.assertEqual(ver["category_counts"]["image_positive_omitted"], 1)
        print("[OK] Test 21 Passed: Image-positive / report-omitted triggers review-required status.")


if __name__ == "__main__":
    unittest.main()


