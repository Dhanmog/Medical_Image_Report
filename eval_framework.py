import os
import sys
import json
import re
from typing import Dict, List, Any, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_refinement import (
    refine_report,
    clean_raw_report,
    _is_term_negated,
    _get_laterality,
    KNOWN_FINDINGS
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "evaluation_dataset.json")


class Phase3EvaluationEngine:
    """
    Quantitative Evaluation Framework for CHEST AI Phase 3.
    Evaluates report quality, clinical finding preservation, negation, laterality,
    severity, device stability, and hallucination rates across 4 processing tiers.
    """

    def __init__(self, dataset_file: str = DATASET_PATH):
        self.dataset_file = dataset_file
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Evaluation dataset file not found at {self.dataset_file}")
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_findings_set(text: str) -> set:
        """Extracts medical findings present in text using known entity matching."""
        if not text:
            return set()
        text_lower = text.lower()
        extracted = set()

        finding_map = {
            "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged", "big heart"],
            "pleural effusion": ["pleural effusion", "fluid around the lung", "fluid around lungs", "pleural fluid", "fluid in pleural space"],
            "atelectasis": ["atelectasis", "partial lung collapse", "collapsed lung", "atelectatic"],
            "pneumothorax": ["pneumothorax", "air around the lung"],
            "pulmonary edema": ["pulmonary edema", "fluid accumulation in the lungs", "fluid in the lungs"],
            "pulmonary vascular congestion": ["pulmonary vascular congestion", "vascular congestion", "congestion in lung blood vessels"],
            "pneumonia": ["pneumonia", "focal pneumonia", "lung infection"],
            "borderline cardiac size": ["borderline cardiac size", "upper limit of normal", "borderline enlarged heart"],
            "chest tube": ["chest tube"],
            "pacemaker": ["pacemaker", "pacer"],
            "sternal wires": ["sternal wires", "sternotomy wires"],
            "central line": ["central line", "catheter", "subclavian", "venous catheter"],
            "decreased lung volumes": ["decreased lung volume", "decreased lung volumes", "low lung volumes"],
            "interval extubation": ["extubated", "extubation", "breathing tube has been removed"]
        }

        for std_finding, synonyms in finding_map.items():
            for syn in synonyms:
                if syn in text_lower and not _is_term_negated(text_lower, syn):
                    extracted.add(std_finding)
                    break

        return extracted

    @staticmethod
    def _extract_negatives_set(text: str) -> set:
        """Extracts explicitly negated findings in text."""
        if not text:
            return set()
        text_lower = text.lower()
        negatives = set()
        targets = ["pneumonia", "pneumothorax", "effusion", "cardiomegaly", "edema", "consolidation"]
        for target in targets:
            if _is_term_negated(text_lower, target):
                negatives.add(target)
        return negatives

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single benchmark case across processing tiers."""
        raw_text = case["raw_transformer_report"]
        preds = case.get("densenet_predictions", {})
        gt_findings = set(case.get("ground_truth_findings", []))
        gt_negatives = set(case.get("ground_truth_negatives", []))
        exp_lat = case.get("expected_laterality", {})
        exp_devices = case.get("expected_devices", [])
        exp_severity = case.get("expected_severity", {})

        # Run master refinement pipeline
        refined = refine_report(raw_text, pathology_predictions=preds)

        prof_text = refined["medical_report"]["full_text"]
        simple_text = refined["simple_report"]

        # Extracted findings from generated report
        gen_findings = self._extract_findings_set(prof_text)
        gen_negatives = self._extract_negatives_set(prof_text)

        # 1. Finding-Level Precision, Recall, F1
        tp = len(gt_findings.intersection(gen_findings))
        fp = len(gen_findings - gt_findings)
        fn = len(gt_findings - gen_findings)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gt_findings and not gen_findings else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gt_findings else 0.0)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else (1.0 if precision == 1.0 and recall == 1.0 else 0.0)

        # 2. Negation Preservation F1
        neg_tp = len(gt_negatives.intersection(gen_negatives))
        neg_fp = len(gen_negatives - gt_negatives)
        neg_fn = len(gt_negatives - gen_negatives)
        neg_prec = neg_tp / (neg_tp + neg_fp) if (neg_tp + neg_fp) > 0 else (1.0 if not gt_negatives and not gen_negatives else 0.0)
        neg_rec = neg_tp / (neg_tp + neg_fn) if (neg_tp + neg_fn) > 0 else (1.0 if not gt_negatives else 0.0)
        neg_f1 = (2 * neg_prec * neg_rec) / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else (1.0 if neg_prec == 1.0 and neg_rec == 1.0 else 0.0)

        # 3. Laterality Preservation
        prof_lower = prof_text.lower()
        lat_correct = 0
        lat_total = len(exp_lat)
        for entity, expected_side in exp_lat.items():
            if expected_side in prof_lower and entity in prof_lower:
                lat_correct += 1
        lat_acc = (lat_correct / lat_total) if lat_total > 0 else 1.0

        # 4. Device Description Score
        device_correct = 0
        device_total = len(exp_devices)
        for dev in exp_devices:
            dev_keywords = dev.lower().split()
            if all(kw in prof_lower for kw in dev_keywords):
                device_correct += 1
        device_score = (device_correct / device_total) if device_total > 0 else 1.0

        # 5. Hallucinations vs Unsupported Findings
        val_res = refined["validation"]
        hallucinations = val_res["details"].get("hallucinations_detected", [])
        is_hallucinating = len(hallucinations) > 0

        # 6. Multi-View Deduplication Efficiency
        raw_sentences = [s.strip() for s in re.split(r"[.\n]", raw_text) if len(s.strip()) > 5]
        prof_sentences = [s.strip() for s in re.split(r"[.\n]", prof_text) if len(s.strip()) > 5]
        dedup_ratio = 1.0 - (len(prof_sentences) / float(len(raw_sentences))) if len(raw_sentences) > 0 else 0.0
        dedup_ratio = max(0.0, dedup_ratio)

        return {
            "case_id": case["id"],
            "title": case["title"],
            "dataset_type": case.get("dataset_type", "Synthetic Benchmark Reference Case"),
            "engine_used": refined["engine_used"],
            "is_llm_active": refined["is_llm_active"],
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "negation_f1": round(neg_f1, 4),
                "laterality_accuracy": round(lat_acc, 4),
                "device_score": round(device_score, 4),
                "hallucination_count": len(hallucinations),
                "is_hallucinating": is_hallucinating,
                "deduplication_ratio": round(dedup_ratio, 4)
            },
            "ground_truth_findings": list(gt_findings),
            "generated_findings": list(gen_findings),
            "generated_negatives": list(gen_negatives),
            "hallucinations_detected": hallucinations
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Runs evaluation over all cases in dataset and calculates aggregate metrics."""
        results = []
        for case in self.dataset:
            res = self.evaluate_case(case)
            results.append(res)

        avg_prec = sum(r["metrics"]["precision"] for r in results) / len(results)
        avg_rec = sum(r["metrics"]["recall"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1_score"] for r in results) / len(results)
        avg_neg_f1 = sum(r["metrics"]["negation_f1"] for r in results) / len(results)
        avg_lat = sum(r["metrics"]["laterality_accuracy"] for r in results) / len(results)
        avg_dev = sum(r["metrics"]["device_score"] for r in results) / len(results)
        avg_dedup = sum(r["metrics"]["deduplication_ratio"] for r in results) / len(results)
        total_hallucinations = sum(r["metrics"]["hallucination_count"] for r in results)

        summary = {
            "evaluation_title": "CHEST AI Phase 3 Quantitative Report Quality Benchmark",
            "dataset_type": "Synthetic / Software Benchmark Reference Dataset (10 Test Cases)",
            "disclaimer": "DISCLAIMER: Benchmark metrics generated on software reference cases for evaluation and documentation purposes. Requires expert clinical validation before diagnostic deployment.",
            "cases_evaluated": len(results),
            "aggregate_metrics": {
                "mean_precision": round(avg_prec, 4),
                "mean_recall": round(avg_rec, 4),
                "mean_f1_score": round(avg_f1, 4),
                "mean_negation_f1": round(avg_neg_f1, 4),
                "mean_laterality_accuracy": round(avg_lat, 4),
                "mean_device_accuracy": round(avg_dev, 4),
                "mean_deduplication_efficiency": round(avg_dedup, 4),
                "total_hallucinations_detected": total_hallucinations
            },
            "case_results": results
        }

        # Save summary JSON
        summary_path = os.path.join(os.path.dirname(__file__), "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    engine = Phase3EvaluationEngine()
    sum_data = engine.run_full_benchmark()
    print("=======================================================")
    print("CHEST AI PHASE 3 QUANTITATIVE EVALUATION COMPLETED")
    print("=======================================================")
    print(json.dumps(sum_data["aggregate_metrics"], indent=2))
