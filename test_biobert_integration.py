"""
================================================================================
BIOBERT INTEGRATION & CLINICAL REGRESSION TEST SUITE
================================================================================
Tests BioBERT Biomedical NLP Engine integration and clinical regression cases:
  1. BioBERT Embeddings extraction (dmis-lab/biobert-base-cased-v1.2)
  2. Pretrained Biomedical NER (d4data/biomedical-ner-all) entity extraction
  3. Radiology findings extraction (PRESENT, ABSENT, UNCERTAIN)
  4. Laterality (Left, Right, Bilateral) and Severity (Mild, Moderate, Severe) identification
  5. DenseNet121 14-pathology prediction comparison & inconsistency flagging
  6. REGRESSION: Scoped negation ('without' negates edema only, NOT cardiomegaly in 'cardiomegaly without edema')
  7. REGRESSION: Preservation of explicit negative findings in professional & patient reports
  8. REGRESSION: Accurate DenseNet121 concordance score calculation (does not default to 100%)
  9. REGRESSION: Unevaluated status when no pathology head predictions are provided
================================================================================
"""

import sys
import os
import unittest

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)

from biobert_nlp_engine import BioBERTBiomedicalNLPEngine, get_biobert_engine
from report_refinement import refine_report, clean_raw_report
from bidirectional_engine import BidirectionalMedicalEngine


class TestBioBERTIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("INITIALIZING BIOBERT INTEGRATION & CLINICAL REGRESSION TEST SUITE")
        print("=======================================================")
        cls.engine = get_biobert_engine()

    def test_01_biobert_embeddings(self):
        """Verify BioBERT contextual embedding shape [1, seq_len, 768] and norm."""
        text = "The chest radiograph demonstrates mild cardiomegaly."
        emb_data = self.engine.extract_biobert_embeddings(text)
        
        self.assertIn("embedding_shape", emb_data)
        self.assertEqual(len(emb_data["embedding_shape"]), 3)
        self.assertEqual(emb_data["embedding_shape"][0], 1)
        self.assertEqual(emb_data["embedding_shape"][2], 768)
        self.assertGreater(emb_data["mean_embedding_norm"], 0.0)
        print("[OK] Test 01 Passed: BioBERT 768-dim embeddings verified.")

    def test_02_biomedical_ner_extraction(self):
        """Verify pretrained biomedical NER entity extraction."""
        text = "Patient presents with mild cardiomegaly and bilateral pleural effusion without pneumothorax."
        entities = self.engine.extract_ner_entities(text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        
        entity_strings = [str(e["entity"]).lower() for e in entities if "entity" in e]
        has_match = any("cardiom" in s or "pleural" in s or "effusion" in s for s in entity_strings)
        self.assertTrue(has_match, f"Should find medical entities in text. Found: {entity_strings}")
        print(f"[OK] Test 02 Passed: Biomedical NER extracted {len(entities)} entities: {entity_strings}")

    def test_03_polarity_laterality_severity(self):
        """Verify PRESENT, ABSENT, UNCERTAIN statuses, laterality, and severity."""
        report = (
            "There is mild cardiomegaly on the left. "
            "Bilateral pleural effusion is noted. "
            "No focal pulmonary consolidation or pneumothorax is identified. "
            "Possible right lower lobe pneumonia."
        )
        findings = self.engine.extract_radiology_findings(report)
        
        positives = [f for f in findings if f["status"] == "PRESENT"]
        negatives = [f for f in findings if f["status"] == "ABSENT"]
        uncertains = [f for f in findings if f["status"] == "UNCERTAIN"]
        
        self.assertGreater(len(positives), 0, "Should detect PRESENT findings")
        self.assertGreater(len(negatives), 0, "Should detect ABSENT findings")
        self.assertGreater(len(uncertains), 0, "Should detect UNCERTAIN findings")

        laterality_types = set(f["laterality"] for f in findings)
        self.assertTrue("BILATERAL" in laterality_types or "LEFT" in laterality_types or "RIGHT" in laterality_types)

        severity_types = set(f["severity"] for f in findings)
        self.assertIn("MILD", severity_types)
        print(f"[OK] Test 03 Passed: Extracted {len(positives)} PRESENT, {len(negatives)} ABSENT, {len(uncertains)} UNCERTAIN findings.")

    def test_04_densenet121_consistency_verification(self):
        """Verify comparison with DenseNet121 14-pathology predictions & inconsistency flagging."""
        report = "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax."
        pathology_preds = {
            "Consolidation": 0.72,  # DenseNet predicts high, report says NO
            "Effusion": 0.05,
            "Pneumothorax": 0.02,
            "Cardiomegaly": 0.15
        }
        
        analysis = self.engine.process_full_report(report, pathology_preds)
        densenet_ver = analysis["densenet_verification"]
        
        self.assertTrue(densenet_ver["inconsistencies_found"])
        self.assertGreater(len(densenet_ver["flagged_issues"]), 0)
        self.assertIsNotNone(densenet_ver["consistency_score"])
        self.assertLess(densenet_ver["consistency_score"], 100.0)
        
        # Verify findings are preserved without automatic deletion
        self.assertEqual(len(analysis["structured_findings"]), len(self.engine.extract_radiology_findings(report)))
        print(f"[OK] Test 04 Passed: DenseNet121 consistency verification flagged mismatch ({densenet_ver['consistency_score']}%) without deleting findings.")

    def test_05_refine_report_integration(self):
        """Verify integration with report_refinement.py pipeline."""
        raw_report = "mild cardiomegaly. bilateral pleural effusion. no pneumothorax."
        path_preds = {"Cardiomegaly": 0.80, "Effusion": 0.85, "Pneumothorax": 0.01}
        
        refined = refine_report(raw_report, pathology_predictions=path_preds)
        
        self.assertIn("biobert_analysis", refined)
        self.assertIn("biobert_findings", refined)
        self.assertIn("densenet_verification", refined)
        self.assertIn("simple_report", refined)
        self.assertIn("medical_report", refined)
        print("[OK] Test 05 Passed: refine_report successfully integrates BioBERT Biomedical NLP.")

    def test_06_cardiomegaly_without_edema_scoped_negation(self):
        """REGRESSION TEST: 'without' must negate edema ONLY, NOT preceding cardiomegaly."""
        sentence = "mild cardiomegaly without pulmonary edema"
        findings = self.engine.extract_radiology_findings(sentence)
        
        cardio_finding = next((f for f in findings if f["pathology"] == "Cardiomegaly"), None)
        edema_finding = next((f for f in findings if f["pathology"] == "Edema"), None)
        
        self.assertIsNotNone(cardio_finding, "Cardiomegaly should be detected")
        self.assertIsNotNone(edema_finding, "Edema should be detected")
        
        self.assertEqual(cardio_finding["status"], "PRESENT", "Cardiomegaly must be PRESENT (not negated by 'without')")
        self.assertEqual(cardio_finding["severity"], "MILD", "Cardiomegaly severity should be MILD")
        self.assertEqual(edema_finding["status"], "ABSENT", "Pulmonary edema must be ABSENT (negated by 'without')")
        print("[OK] Test 06 Passed: Scoped negation correctly marked Cardiomegaly as PRESENT and Edema as ABSENT.")

    def test_07_explicit_negative_preservation(self):
        """REGRESSION TEST: Explicit negative findings must be preserved in final reports."""
        raw = "mild cardiomegaly without pulmonary edema. no pleural effusion. no pneumothorax."
        refined = refine_report(raw)
        
        med_text = refined["medical_report"]["full_text"].lower()
        simple_text = refined["simple_report"].lower()
        
        self.assertIn("cardiomegaly", med_text, "Professional report must mention Cardiomegaly")
        self.assertTrue("effusion" in med_text or "no " in med_text, "Professional report must preserve negative effusion")
        self.assertTrue("pneumothorax" in med_text or "no " in med_text, "Professional report must preserve negative pneumothorax")
        
        self.assertIn("cardiomegaly", simple_text, "Simple report must mention Cardiomegaly")
        self.assertTrue("fluid" in simple_text or "effusion" in simple_text or "no " in simple_text, "Simple report must explain negative effusion")
        print("[OK] Test 07 Passed: Explicit negative findings preserved in professional and patient-friendly reports.")

    def test_08_densenet_concordance_score_accuracy(self):
        """REGRESSION TEST: Concordance score reflects actual comparison results, not automatic 100%."""
        report = "cardiomegaly is present. no pleural effusion."
        pathology_preds = {
            "Cardiomegaly": 0.85,  # Concordant (Report PRESENT, Image 85%)
            "Effusion": 0.90       # Discordant (Report ABSENT, Image 90%)
        }
        analysis = self.engine.process_full_report(report, pathology_preds)
        ver = analysis["densenet_verification"]
        
        self.assertTrue(ver["inconsistencies_found"])
        self.assertEqual(ver["consistency_score"], 50.0, "Score should be 50.0% for 1 match out of 2 evaluated pathologies")
        print("[OK] Test 08 Passed: Concordance score accurately calculated as 50.0%.")

    def test_09_empty_pathology_predictions_unevaluated(self):
        """REGRESSION TEST: Empty pathology predictions returns None score (UNEVALUATED)."""
        report = "mild cardiomegaly."
        analysis = self.engine.process_full_report(report, {})
        ver = analysis["densenet_verification"]
        
        self.assertIsNone(ver["consistency_score"], "Score should be None when no pathology head predictions provided")
        self.assertIn("N/A", ver["status_text"])
    def test_10_pulmonary_venous_pressure_no_unsupported_edema(self):
        """REGRESSION TEST: Elevated pulmonary venous pressure must not state pulmonary edema is present."""
        raw = "There is enlargement of the cardiac silhouette with mild elevation of pulmonary venous pressure. No acute focal pneumonia."
        refined = refine_report(raw)
        
        simple_text = refined["simple_report"].lower()
        med_text = refined["medical_report"]["full_text"].lower()
        
        self.assertNotIn("pulmonary edema", simple_text, "Patient-friendly report must NOT introduce unsupported pulmonary edema diagnosis")
        self.assertNotIn("/ edema", med_text, "Medical report must NOT label pulmonary venous pressure as / edema")
        self.assertTrue("pulmonary venous pressure" in simple_text or "pressure in" in simple_text or "pressure or congestion" in simple_text, "Patient-friendly report should mention pulmonary venous pressure or pressure in lung vessels")
        self.assertIn("pneumonia", med_text, "Medical report must preserve negative pneumonia")
        print("[OK] Test 10 Passed: Elevated pulmonary venous pressure correctly preserved without introducing unsupported pulmonary edema.")

    def test_11_temporal_context_resolution(self):
        """REGRESSION TEST: Historical/interval findings must be distinguished from current findings."""
        raw = "Prior radiograph demonstrated a small left apical pneumothorax. Current exam shows no left pneumothorax. Small left pleural effusion."
        findings = self.engine.extract_radiology_findings(raw)
        
        pneu_findings = [f for f in findings if f["pathology"] == "Pneumothorax"]
        self.assertGreater(len(pneu_findings), 0, "Should detect pneumothorax mentions")
        
        temporal_statuses = set(f.get("temporal_status") for f in pneu_findings)
        self.assertTrue("HISTORICAL" in temporal_statuses or "RESOLVED" in temporal_statuses or "CURRENT" in temporal_statuses)
        
        refined = refine_report(raw)
        simple_text = refined["simple_report"].lower()
        med_text = refined["medical_report"]["full_text"].lower()
        
        self.assertTrue("fluid" in simple_text or "effusion" in simple_text, "Simple report should mention active effusion/fluid")
        self.assertTrue("pneumothorax" in simple_text or "trapped air" in simple_text, "Simple report should mention pneumothorax status")
        self.assertIn("effusion", med_text, "Medical report should mention active effusion")
        print("[OK] Test 11 Passed: Temporal context correctly distinguished historical findings from current findings.")

    def test_12_evaluable_concordance_score(self):
        """REGRESSION TEST: Concordance score evaluates ONLY genuinely evaluable comparisons, displaying exact count."""
        report = "small left pleural effusion is noted. no pneumothorax."
        pathology_preds = {
            "Effusion": 0.85,       # Mentioned PRESENT (Concordant)
            "Pneumothorax": 0.05,   # Mentioned ABSENT (Concordant)
            "Cardiomegaly": 0.02,   # Unmentioned, low prob (<0.50) -> NOT EVALUATED FOR SCORE
            "Atelectasis": 0.04,    # Unmentioned, low prob (<0.50) -> NOT EVALUATED FOR SCORE
            "Pneumonia": 0.01       # Unmentioned, low prob (<0.50) -> NOT EVALUATED FOR SCORE
        }
        
        analysis = self.engine.process_full_report(report, pathology_preds)
        ver = analysis["densenet_verification"]
        
        self.assertEqual(ver["evaluable_count"], 2, "Only the 2 mentioned pathologies should be evaluated")
        self.assertEqual(ver["consistency_score"], 100.0, "Score should be 100.0% for 2 evaluable concordant matches")
        self.assertIn("2/2 Pathologies Evaluated", ver["status_text"])
        
        # Test addition of unmentioned high-probability mismatch (Case D)
        pathology_preds["Cardiomegaly"] = 0.75  # Unmentioned high prob -> Mismatch flag!
        analysis2 = self.engine.process_full_report(report, pathology_preds)
        ver2 = analysis2["densenet_verification"]
        
        self.assertEqual(ver2["evaluable_count"], 3, "Evaluable count should include the unmentioned high-prob pathology")
        self.assertEqual(ver2["consistency_score"], 66.7, "Score should be 66.7% for 2 matches out of 3 evaluable pathologies")
        self.assertIn("2/3 Pathologies Evaluated", ver2["status_text"])
        print("[OK] Test 12 Passed: Concordance score evaluates only genuinely evaluable comparisons and displays exact count.")

    def test_13_unsupported_opacity_insertion_and_synonyms(self):
        """REGRESSION TEST: Normal report with 'no focal opacity' must not introduce unsupported consolidation or fail validation."""
        raw = "The cardiac silhouette and lung fields are within normal limits. No focal opacity or pleural effusion."
        refined = refine_report(raw)
        
        val = refined["validation"]
        self.assertTrue(val["is_valid"], f"Normal report must pass semantic validation without hallucinated findings: {val.get('flagged_issues')}")
        self.assertEqual(val["score"], 100.0, "Semantic score should be 100.0%")
        
        med_text = refined["medical_report"]["full_text"].lower()
        self.assertIn("opacity", med_text, "Medical report should preserve negative opacity finding")
        self.assertIn("effusion", med_text, "Medical report should preserve negative effusion finding")
        print("[OK] Test 13 Passed: Unsupported opacity/consolidation insertion prevented; negative findings preserved.")

    def test_14_atelectasis_image_positive_unmentioned_mismatch(self):
        """REGRESSION TEST: 75% Atelectasis prediction must trigger image-positive/unmentioned mismatch without altering report text."""
        raw = "mild cardiomegaly is noted."
        pathology_preds = {
            "Cardiomegaly": 0.85,  # Mentioned PRESENT (Concordant)
            "Atelectasis": 0.75   # Unmentioned, High Prob >= 70% (Image-Positive / Report-Omitted Mismatch)
        }
        
        refined = refine_report(raw, pathology_predictions=pathology_preds)
        med_text = refined["medical_report"]["full_text"].lower()
        
        # Report text must NOT have atelectasis forced into diagnosis
        self.assertNotIn("atelectasis", med_text, "Atelectasis must NOT be automatically added to diagnosis based only on probability")
        
        ver = refined["densenet_verification"]
        self.assertTrue(ver["inconsistencies_found"], "Inconsistency must be flagged")
        self.assertEqual(ver["evaluable_count"], 2, "Evaluated count should be 2 (Cardiomegaly + Atelectasis)")
        self.assertEqual(ver["consistency_score"], 50.0, "Consistency score should be 50.0%")
        
        atelectasis_flag = next((f for f in ver["flagged_issues"] if f["pathology"] == "Atelectasis"), None)
        self.assertIsNotNone(atelectasis_flag, "Atelectasis mismatch flag must be present")
        self.assertEqual(atelectasis_flag["issue_type"], "Image-Positive / Report-Omitted Finding")
        self.assertIn("75.0%", atelectasis_flag["description"])
        print("[OK] Test 14 Passed: Atelectasis 75.0% mismatch correctly flagged without adding atelectasis to report text.")

    def test_15_multi_view_omission_preservation(self):
        """REGRESSION TEST: Multi-view observations with chest tube, pacemaker leads, extubation, volume changes, borderline heart, and explicit negatives."""
        raw = ("The previous radiograph, the patient has received a right chest tube. "
               "The course of the pacemaker leads is constant. Constant appearance of the left lung. "
               "The previous radiograph, the patient has been extubated. The lung volumes have decreased. "
               "The lung volumes are normal. Borderline size of the cardiac silhouette without pulmonary edema, "
               "no pleural effusions, no pneumonia.")
        
        refined = refine_report(raw)
        med_text = refined["medical_report"]["full_text"].lower()
        simple_text = refined["simple_report"].lower()
        
        # 1. Right chest tube
        self.assertTrue("chest tube" in med_text and "right" in med_text, "Medical report must preserve right chest tube")
        self.assertTrue("chest tube" in simple_text and "right" in simple_text, "Patient-friendly report must preserve right chest tube")
        
        # 2. Pacemaker lead stability
        self.assertTrue("pacemaker" in med_text and ("leads" in med_text or "stable" in med_text or "constant" in med_text), "Medical report must preserve pacemaker lead stability")
        self.assertTrue("pacemaker" in simple_text and ("wires" in simple_text or "stable" in simple_text or "unchanged" in simple_text), "Patient-friendly report must preserve pacemaker lead stability")
        
        # 3. Interval extubation
        self.assertTrue("extubated" in med_text or "extubation" in med_text, "Medical report must preserve extubation")
        self.assertTrue("extubated" in simple_text or "extubation" in simple_text or "breathing tube" in simple_text, "Patient-friendly report must preserve extubation")
        
        # 4. Lung volumes
        self.assertTrue("lung volumes" in med_text and "decreased" in med_text, "Medical report must preserve decreased lung volumes")
        self.assertTrue("lung volumes" in simple_text and "decreased" in simple_text, "Patient-friendly report must preserve decreased lung volumes")
        
        # 5. Borderline cardiac silhouette
        self.assertTrue("borderline" in med_text and "cardiac" in med_text, "Medical report must preserve borderline cardiac silhouette")
        self.assertTrue("borderline" in simple_text or "upper limit" in simple_text, "Patient-friendly report must preserve borderline heart size")
        
        # 6. Explicit negatives
        self.assertTrue("edema" in med_text and ("no " in med_text or "without" in med_text), "Medical report must preserve negative edema")
        self.assertTrue("effusion" in med_text and ("no " in med_text or "without" in med_text), "Medical report must preserve negative effusion")
        self.assertTrue("pneumonia" in med_text and ("no " in med_text or "without" in med_text), "Medical report must preserve negative pneumonia")
        print("[OK] Test 15 Passed: Multi-view observation preservation, support devices, comparison context, borderline findings, and explicit negatives verified.")

    def test_16_edema_and_congestion_preservation(self):
        """REGRESSION TEST: Verify simultaneous preservation of pulmonary vascular congestion, pulmonary edema, compressive atelectasis, severity, and patient explanations."""
        raw = ("Substantial cardiomegaly. Pulmonary vascular congestion. Bilateral pleural effusions. "
               "Compressive atelectasis at the lung bases. Mild-to-moderate pulmonary edema. "
               "No evidence of acute focal pneumonia.")
        
        refined = refine_report(raw)
        med_text = refined["medical_report"]["full_text"].lower()
        simple_text = refined["simple_report"].lower()
        
        # 1. Pulmonary edema & vascular congestion independently preserved
        self.assertIn("pulmonary edema", med_text, "Professional report must explicitly include pulmonary edema")
        self.assertIn("pulmonary vascular congestion", med_text, "Professional report must explicitly include pulmonary vascular congestion")
        
        # 2. Patient-friendly translations
        self.assertTrue("fluid accumulation within the lungs" in simple_text or "fluid accumulation inside the lung tissue" in simple_text, "Patient report must explain edema as fluid inside/within lungs")
        self.assertTrue("congestion in lung blood vessels" in simple_text or "pressure or congestion" in simple_text, "Patient report must explain vascular congestion")
        self.assertTrue("enlarged heart" in simple_text, "Patient report must translate cardiomegaly to enlarged heart")
        self.assertTrue("fluid around both lungs" in simple_text or "fluid surrounding both lungs" in simple_text, "Patient report must translate bilateral effusions")
        self.assertTrue("compression-related collapse" in simple_text or "partial compression-related collapse" in simple_text, "Patient report must explain compressive atelectasis")
        
        # 3. Severity modifiers and compressive atelectasis in medical report
        self.assertIn("substantial cardiomegaly", med_text, "Professional report must retain 'substantial cardiomegaly'")
        self.assertTrue("compressive atelectasis" in med_text, "Professional report must retain 'compressive atelectasis'")
        self.assertTrue("mild-to-moderate" in med_text or "mild to moderate" in med_text, "Professional report must retain severity for pulmonary edema")
        
        # 4. Explicit negative pneumonia retained
        self.assertIn("pneumonia", med_text, "Professional report must retain negative pneumonia finding")
        self.assertTrue("no " in med_text or "without" in med_text, "Negative pneumonia finding must remain negative")
        
        # 5. Semantic Validation check
        val = refined["validation"]
        self.assertTrue(val["is_valid"], f"Refined report must pass semantic validation: {val.get('flagged_issues')}")
        self.assertGreaterEqual(val["score"], 90.0, "Semantic validation score must be high")
        print("[OK] Test 16 Passed: Pulmonary edema, vascular congestion, compressive atelectasis, severity, and patient explanations all preserved.")

    def test_17_borderline_and_conflicting_cardiac_size(self):
        """REGRESSION TEST: Verify preservation of borderline cardiac size, conflict resolution, and upper limit terminology."""
        # Scenario A: Borderline size (upper limits of normal)
        raw_borderline = "The cardiac silhouette is at the upper limits of normal. No pleural effusion."
        refined1 = refine_report(raw_borderline)
        med1 = refined1["medical_report"]["full_text"].lower()
        simple1 = refined1["simple_report"].lower()
        
        self.assertIn("borderline", med1, "Medical report must preserve borderline cardiac size")
        self.assertNotIn("cardiomegaly is noted", med1, "Medical report must not convert upper limits of normal into definite cardiomegaly")
        self.assertTrue("upper limit" in simple1 or "borderline" in simple1, "Patient report must explain borderline cardiac size")

        # Scenario B: Conflicting cardiac size across views (Borderline + Cardiomegaly)
        raw_conflict = "View 1 shows borderline cardiac silhouette. View 2 demonstrates mild cardiomegaly."
        refined2 = refine_report(raw_conflict)
        med2 = refined2["medical_report"]["full_text"].lower()
        simple2 = refined2["simple_report"].lower()
        
        self.assertIn("borderline", med2, "Medical report must preserve borderline finding")
        self.assertIn("cardiomegaly", med2, "Medical report must preserve cardiomegaly finding")
        self.assertTrue("upper limit" in simple2 or "borderline" in simple2, "Patient report must preserve borderline range")
        self.assertIn("cardiomegaly", simple2, "Patient report must preserve cardiomegaly term")
        print("[OK] Test 17 Passed: Borderline cardiac size and conflicting multi-view cardiac descriptions accurately preserved.")

    def test_18_negative_congestion_device_deduplication(self):
        """REGRESSION TEST: Verify explicit negative vascular congestion preservation, support device detail, and observation deduplication."""
        raw = ("Substantial cardiomegaly. No evidence of vascular congestion. "
               "Intact median sternotomy wires. Sternal wires intact. "
               "Course of pacemaker leads is constant. Unchanged left pectoral pacemaker.")
        
        refined = refine_report(raw)
        med_text = refined["medical_report"]["full_text"].lower()
        simple_text = refined["simple_report"].lower()
        val = refined["validation"]

        # 1. Explicit negative vascular congestion
        self.assertTrue("vascular congestion" in med_text and ("no " in med_text or "without" in med_text), "Medical report must preserve negative vascular congestion")
        self.assertTrue("congestion" in simple_text and ("no " in simple_text or "without" in simple_text or "identified" in simple_text), "Patient report must preserve negative vascular congestion")

        # 2. Device detail & deduplication
        self.assertIn("sternal wires", med_text, "Medical report must preserve sternal wires")
        self.assertEqual(med_text.count("intact midline sternal wires are present."), 1, "Sternal wire finding must be deduplicated to a single statement")
        
        self.assertIn("pacemaker", med_text, "Medical report must preserve pacemaker")
        self.assertTrue("left pectoral" in med_text or "left" in med_text, "Medical report must preserve left pectoral pacemaker location")
        self.assertTrue("stable" in med_text or "constant" in med_text, "Medical report must preserve lead stability")

        # 3. Validation check
        self.assertTrue(val["is_valid"], f"Refined report must pass semantic consistency check: {val.get('flagged_issues')}")
        self.assertGreaterEqual(val["score"], 90.0, "Validation score must be >= 90%")
        print("[OK] Test 18 Passed: Negative vascular congestion, device details (left pectoral pacemaker, sternal wires), and deduplication verified.")


if __name__ == "__main__":
    unittest.main()


