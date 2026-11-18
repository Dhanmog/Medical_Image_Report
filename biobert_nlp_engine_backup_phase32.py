"""
================================================================================
BIOBERT BIOMEDICAL NLP & RADIOLOGY FINDINGS ENGINE
================================================================================
This module implements a modular biomedical NLP engine using:
  1. Checkpoint: dmis-lab/biobert-base-cased-v1.2 (BioBERT contextual embeddings)
  2. Pretrained Biomedical NER: d4data/biomedical-ner-all (Medical entity extraction)
  3. Radiology Findings Extraction: Positives (PRESENT), Negatives (ABSENT), Uncertainties (UNCERTAIN), Laterality, Severity
  4. Scoped Clause-Aware Negation Parsing (e.g. 'without' negates following finding only)
  5. DenseNet121 14-Pathology Prediction Consistency Verification & Score Calculation
  6. Non-Destructive Inconsistency Flagging (without deleting findings) & LLM Refinement Integration
================================================================================
"""

import re
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModel, pipeline

# BioBERT Pretrained Checkpoint
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
NER_MODEL_NAME = "d4data/biomedical-ner-all"

# Standard 14 CheXpert Pathology Classes + Support Devices & Normal
CHEXPERT_PATHOLOGIES = [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Pneumonia",
    "Pleural Thickening",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Opacity",
    "Support Devices",
]

PATHOLOGY_KEYWORD_MAP = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged", "big heart", "cardiac silhouette", "cardiac size is enlarged", "enlarged cardiac silhouette", "borderline size of the cardiac silhouette", "borderline cardiac"],
    "Pleural Effusion": ["pleural effusion", "effusion", "effusions", "fluid around", "fluid in pleural", "pleural fluid"],
    "Atelectasis": ["atelectasis", "collapsed lung", "lung collapse", "atelectatic"],
    "Pneumothorax": ["pneumothorax", "pneumothoraces", "air around", "air in pleural"],
    "Pneumonia": ["pneumonia", "lung infection", "focal pneumonia"],
    "Edema": ["edema", "pulmonary edema", "vascular congestion", "pulmonary congestion", "venous pressure"],
    "Opacity": ["opacity", "opacities", "cloudy spot", "hazy area", "infiltrate", "air-space opacity", "airspace opacity"],
    "Consolidation": ["consolidation", "airspace consolidation", "focal consolidation"],
    "Support Devices": ["pacemaker", "pacer", "catheter", "sternal wires", "sternotomy", "tube", "line", "chest tube", "extubated", "extubation", "intubated", "leads"],
    "Lung Volume": ["lung volumes", "lung volume", "low lung volumes", "decreased lung volumes", "hypoinflation"],
    "Normal": ["normal limits", "unremarkable", "clear lungs", "no acute cardiopulmonary", "normal heart size"],
}


def normalize_pathology_key(key: str) -> str:
    """Normalizes pathology names to canonical key string for consistent cross-module comparison."""
    if not key:
        return ""
    k = key.lower().strip().replace(" ", "_").replace("-", "_")
    mapping = {
        "pleural_effusion": "effusion",
        "effusion": "effusion",
        "pleural_fluid": "effusion",
        "pulmonary_edema": "edema",
        "edema": "edema",
        "vascular_congestion": "edema",
        "lung_opacity": "opacity",
        "airspace_opacity": "opacity",
        "opacity": "opacity",
        "focal_pneumonia": "pneumonia",
        "pneumonia": "pneumonia",
        "cardiomegaly": "cardiomegaly",
        "enlarged_cardiomediastinum": "cardiomegaly",
        "enlarged_heart": "cardiomegaly",
        "atelectasis": "atelectasis",
        "pneumothorax": "pneumothorax",
        "consolidation": "consolidation",
        "support_devices": "support_devices",
        "pacemaker": "support_devices",
        "pacer": "support_devices"
    }
    return mapping.get(k, k)


class BioBERTBiomedicalNLPEngine:
    """
    Modular BioBERT NLP engine for medical report analysis, biomedical NER entity
    extraction, polarity/certainty/laterality classification, and DenseNet121 consistency verification.
    """

    def __init__(self, biobert_model_name: str = BIOBERT_MODEL_NAME, ner_model_name: str = NER_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.biobert_model_name = biobert_model_name
        self.ner_model_name = ner_model_name

        self.tokenizer = None
        self.biobert_model = None
        self.ner_pipeline = None

        self._load_biobert()
        self._load_ner()

    def _load_biobert(self):
        """Loads BioBERT tokenizer and PyTorch model weights."""
        try:
            print(f"[BioBERT Engine] Loading BioBERT model: {self.biobert_model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.biobert_model_name)
            self.biobert_model = AutoModel.from_pretrained(self.biobert_model_name)
            self.biobert_model.to(self.device)
            self.biobert_model.eval()
            print("[BioBERT Engine] BioBERT loaded successfully.")
        except Exception as e:
            print(f"[BioBERT Engine] Warning: Could not load BioBERT model ({e}). Will use fallback.")

    def _load_ner(self):
        """Loads pretrained biomedical NER pipeline."""
        try:
            print(f"[BioBERT Engine] Loading Biomedical NER pipeline: {self.ner_model_name}...")
            device_id = 0 if self.device.type == "cuda" else -1
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
                device=device_id
            )
            print("[BioBERT Engine] Biomedical NER loaded successfully.")
        except Exception as e:
            print(f"[BioBERT Engine] Warning: Could not load NER pipeline ({e}). Will use BioBERT pattern matching.")

    def extract_biobert_embeddings(self, text: str) -> Dict[str, Any]:
        """
        Processes medical text through BioBERT to produce 768-dimensional contextual token embeddings.
        Returns tensor shape, pooled embedding representation, and token count.
        """
        if not text or not text.strip():
            return {
                "token_count": 0,
                "embedding_shape": [0, 768],
                "cls_embedding": np.zeros(768).tolist(),
                "mean_embedding_norm": 0.0
            }

        if self.tokenizer is None or self.biobert_model is None:
            return {
                "token_count": len(text.split()),
                "embedding_shape": [1, len(text.split()), 768],
                "cls_embedding": np.zeros(768).tolist(),
                "mean_embedding_norm": 1.0
            }

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.biobert_model(**inputs)

            embeddings = outputs.last_hidden_state
            seq_len = inputs["input_ids"].shape[1]

            cls_emb = embeddings[0, 0, :].cpu().numpy().tolist()
            mean_emb = torch.mean(embeddings[0], dim=0).cpu().numpy()
            norm_val = float(np.linalg.norm(mean_emb))

            return {
                "token_count": seq_len,
                "embedding_shape": list(embeddings.shape),
                "cls_embedding": cls_emb,
                "mean_embedding_norm": round(norm_val, 4)
            }
        except Exception as e:
            print(f"[BioBERT Engine] Embedding extraction error: {e}")
            return {
                "token_count": 0,
                "embedding_shape": [0, 768],
                "cls_embedding": np.zeros(768).tolist(),
                "mean_embedding_norm": 0.0
            }

    def extract_ner_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts biomedical entities using the pretrained Biomedical NER pipeline.
        Cleans subword tokens (##) and deduplicates extracted entities.
        """
        if not text or not self.ner_pipeline:
            return []

        try:
            raw_entities = self.ner_pipeline(text)
            formatted = []
            seen_terms = set()

            for ent in raw_entities:
                word = ent.get("word", "").replace("##", "").strip()
                group = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                score = float(ent.get("score", 0.0))
                
                key = (word.lower(), group)
                if word and len(word) > 1 and key not in seen_terms:
                    seen_terms.add(key)
                    formatted.append({
                        "entity": word,
                        "category": group,
                        "confidence": round(score, 4)
                    })
            return formatted
        except Exception as e:
            print(f"[BioBERT Engine] NER extraction error: {e}")
            return []

    def is_finding_negated_in_sentence(self, sentence: str, target_keywords: List[str]) -> bool:
        """
        Strict scoped clause-aware negation matching.
        Ensures triggers like 'without' or 'no' only negate target findings that follow them
        or fall within their immediate syntactic clause, without improperly negating preceding findings.
        """
        s_lower = sentence.lower().strip()

        # 1. Post-negation patterns (target finding followed by negation phrase)
        # e.g., "pleural effusion is absent", "pneumothorax is not identified", "cardiomegaly is ruled out"
        for kw in target_keywords:
            pattern_post = r"\b" + re.escape(kw.lower()) + r"s?\s+(is\s+|are\s+)?(absent|not\s+seen|not\s+identified|not\s+observed|ruled\s+out|negative)\b"
            if re.search(pattern_post, s_lower):
                return True

        # 2. Pre-negation triggers
        neg_triggers = [
            r"\bno\b", r"\bwithout\b", r"\babsent\b", r"\bnegative for\b",
            r"\bfree of\b", r"\bdenies\b", r"\bno evidence of\b", r"\bclear of\b",
            r"\bunremarkable for\b"
        ]

        for kw in target_keywords:
            kw_re = r"\b" + re.escape(kw.lower()) + r"s?\b"
            for trig in neg_triggers:
                # Matches negation trigger followed by 0-5 words and target keyword
                pattern_pre = trig + r"(?:\s+[\w\-]+){0,5}?\s+" + kw_re
                match = re.search(pattern_pre, s_lower)
                if match:
                    matched_text = match.group(0)
                    # Ensure trigger is not separated by clause conjunctions like 'but' or 'with'
                    if not re.search(r"\b(but|however|although|with|showing|demonstrating)\b", matched_text[len(trig):]):
                        return True

        return False

    def classify_polarity_certainty(self, sentence: str, target_finding: str) -> Tuple[str, str, str]:
        """
        Determines Polarity (POSITIVE, NEGATIVE, UNCERTAIN), Clinical Status (PRESENT, ABSENT, UNCERTAIN),
        and Certainty Level.
        """
        s_lower = sentence.lower().strip()
        keywords = PATHOLOGY_KEYWORD_MAP.get(target_finding, [target_finding.lower()])

        # 1. Scoped Negation Check
        is_negated = self.is_finding_negated_in_sentence(sentence, keywords)

        # 2. Uncertainty Check
        unc_patterns = [
            r"\bpossible\b", r"\bprobable\b", r"\bcannot be excluded\b",
            r"\bborderline\b", r"\bsuspicious for\b", r"\bmay represent\b",
            r"\bequivocal\b", r"\bsuggestive of\b", r"\bquestionable\b",
            r"\bevaluated for\b", r"\bcould represent\b", r"\brule out\b"
        ]
        is_uncertain = any(re.search(pat, s_lower) for pat in unc_patterns)

        # 3. Normal / Unremarkable Observation Check
        norm_patterns = [
            r"\bunremarkable\b", r"\bwithin normal limits\b", r"\bnormal course\b",
            r"\bnormal appearance\b", r"\bintact\b", r"\bunchanged\b", r"\bin position\b",
            r"\bnormal size\b", r"\bclear lungs\b"
        ]
        is_normal = any(re.search(pat, s_lower) for pat in norm_patterns)

        if is_negated:
            return "NEGATIVE", "ABSENT", "CONFIRMED_NEGATIVE"
        elif is_normal and not is_uncertain:
            return "NEUTRAL", "NORMAL", "NORMAL_OBSERVATION"
        elif is_uncertain:
            return "UNCERTAIN", "UNCERTAIN", "POSSIBLE_EQUIVOCAL"
        else:
            return "POSITIVE", "PRESENT", "CONFIRMED_POSITIVE"

    def extract_laterality(self, sentence: str) -> str:
        """Identifies anatomical laterality: LEFT, RIGHT, BILATERAL, or UNSPECIFIED."""
        s_lower = sentence.lower()
        if re.search(r"\b(bilateral|bilaterally|both|both-sided|both bases|both lungs)\b", s_lower):
            return "BILATERAL"
        
        has_left = bool(re.search(r"\b(left|left-sided|left base|left lung|left lower|left upper|left hemithorax)\b", s_lower))
        has_right = bool(re.search(r"\b(right|right-sided|right base|right lung|right lower|right upper|right hemithorax)\b", s_lower))

        if has_left and has_right:
            return "BILATERAL"
        elif has_left:
            return "LEFT"
        elif has_right:
            return "RIGHT"
        return "UNSPECIFIED"

    def extract_severity(self, sentence: str) -> str:
        """Identifies severity grade: MILD, MODERATE, SEVERE, or UNSPECIFIED."""
        s_lower = sentence.lower()
        if re.search(r"\b(mild|slight|small|minor|trace|faint)\b", s_lower):
            return "MILD"
        elif re.search(r"\b(severe|marked|large|massive|gross|prominent|extensive)\b", s_lower):
            return "SEVERE"
        elif re.search(r"\b(moderate|modest|moderate-to-large)\b", s_lower):
            return "MODERATE"
        return "UNSPECIFIED"

    def extract_temporal_status(self, sentence: str) -> str:
        """Identifies temporal context of a finding: CURRENT, HISTORICAL, or RESOLVED."""
        s_lower = sentence.lower()
        if re.search(r"\b(resolved|interval resolution|cleared|disappeared|no longer seen)\b", s_lower):
            return "RESOLVED"
        elif re.search(r"\b(prior|previous|earlier|past|history of|h/o|old|compared to prior|compared with prior)\b", s_lower):
            return "HISTORICAL"
        return "CURRENT"

    def extract_radiology_findings(self, report_text: str) -> List[Dict[str, Any]]:
        """
        Parses report text into structured findings with BioBERT embeddings, NER entities,
        polarity (positive, negative, uncertain), status (PRESENT, ABSENT, UNCERTAIN),
        temporal status (CURRENT, HISTORICAL, RESOLVED), laterality, and severity.
        """
        if not report_text or not report_text.strip():
            return []

        sentences = [s.strip() for s in re.split(r"[.;!\n]", report_text) if s and len(s.strip()) > 1]
        structured_findings = []

        for idx, sentence in enumerate(sentences):
            s_lower = sentence.lower()

            detected_paths = []
            for path_name, keywords in PATHOLOGY_KEYWORD_MAP.items():
                if any(kw in s_lower for kw in keywords):
                    detected_paths.append(path_name)

            if not detected_paths:
                detected_paths = ["General Clinical Observation"]

            laterality = self.extract_laterality(sentence)
            severity = self.extract_severity(sentence)
            temporal_status = self.extract_temporal_status(sentence)

            for path_name in detected_paths:
                polarity, status_val, certainty = self.classify_polarity_certainty(sentence, path_name)
                
                emb_info = self.extract_biobert_embeddings(sentence)
                ner_ents = self.extract_ner_entities(sentence)

                structured_findings.append({
                    "id": len(structured_findings) + 1,
                    "pathology": path_name,
                    "sentence": sentence,
                    "polarity": polarity,
                    "status": status_val,  # PRESENT, ABSENT, or UNCERTAIN
                    "temporal_status": temporal_status,  # CURRENT, HISTORICAL, or RESOLVED
                    "certainty": certainty,
                    "laterality": laterality,
                    "severity": severity,
                    "ner_entities": ner_ents,
                    "biobert_embeddings": {
                        "embedding_shape": emb_info["embedding_shape"],
                        "mean_norm": emb_info["mean_embedding_norm"],
                        "token_count": emb_info["token_count"]
                    }
                })

        return structured_findings

    def compare_with_densenet(
        self,
        extracted_findings: List[Dict[str, Any]],
        pathology_predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compares BioBERT extracted report findings against DenseNet121 14-pathology predictions.
        Calculates ACTUAL concordance percentage using ONLY genuinely evaluable comparisons.
        UNMENTIONED findings with low probability (<0.50) are not counted as confirmed concordant.
        """
        if not pathology_predictions:
            return {
                "consistency_score": None,
                "status_text": "N/A (No pathology head predictions active for selected model)",
                "inconsistencies_found": False,
                "flag_count": 0,
                "flagged_issues": [],
                "comparison_matrix": []
            }

        confirmed_contradictions = []
        image_positive_omitted = []
        text_positive_low_conf = []
        model_uncertainties = []
        concordant_normals = []

        inconsistency_flags = []
        comparison_matrix = []

        report_path_map = {}
        for f in extracted_findings:
            p_key = normalize_pathology_key(f.get("pathology", ""))
            if p_key not in report_path_map:
                report_path_map[p_key] = []
            report_path_map[p_key].append(f)

        concordant_count = 0
        evaluable_count = 0

        for path_name, pred_prob in pathology_predictions.items():
            p_key = normalize_pathology_key(path_name)
            matched_findings = report_path_map.get(p_key, [])

            current_findings = [f for f in matched_findings if f.get("temporal_status", "CURRENT") == "CURRENT"]
            eval_findings = current_findings if current_findings else matched_findings

            has_positive = any(f["status"] == "PRESENT" and f.get("temporal_status", "CURRENT") == "CURRENT" for f in eval_findings)
            has_negative = any(f["status"] == "ABSENT" or f.get("temporal_status") == "RESOLVED" for f in eval_findings)
            has_uncertain = any(f["status"] == "UNCERTAIN" for f in eval_findings)
            has_normal = any(f["status"] == "NORMAL" for f in eval_findings)
            is_mentioned = len(matched_findings) > 0

            status = "UNMENTIONED"
            flag_note = None
            category = "CONCORDANT_NORMAL"
            is_evaluable = False

            # Category 1: Confirmed Contradiction (DenseNet prob >= 0.70, but text explicitly states ABSENT/RESOLVED)
            if pred_prob >= 0.70 and has_negative and not has_positive:
                status = "MISMATCH_CONFIRMED_CONTRADICTION"
                category = "CONFIRMED_CONTRADICTION"
                flag_note = f"CONFIRMED CONTRADICTION: DenseNet121 predicts high likelihood ({pred_prob*100:.1f}%) for '{path_name}', but text explicitly states ABSENT or RESOLVED ('{matched_findings[0]['sentence']}')."
                issue = {
                    "pathology": path_name,
                    "category": category,
                    "issue_type": "Confirmed Contradiction",
                    "severity": "HIGH",
                    "densenet_probability": round(pred_prob, 4),
                    "biobert_report_status": "ABSENT",
                    "description": flag_note
                }
                inconsistency_flags.append(issue)
                confirmed_contradictions.append(issue)
                is_evaluable = True

            # Category 2: Image-Positive / Report-Omitted (High prob >= 0.70, but UNMENTIONED in text)
            elif not is_mentioned and pred_prob >= 0.70:
                status = "MISMATCH_IMAGE_POS_REPORT_OMITTED"
                category = "IMAGE_POS_REPORT_OMITTED"
                flag_note = f"IMAGE-POSITIVE / REPORT-OMITTED: DenseNet121 predicts high likelihood ({pred_prob*100:.1f}%) for '{path_name}', but it is UNMENTIONED in text report."
                issue = {
                    "pathology": path_name,
                    "category": category,
                    "issue_type": "Image-Positive / Report-Omitted Finding",
                    "severity": "MEDIUM",
                    "densenet_probability": round(pred_prob, 4),
                    "biobert_report_status": "UNMENTIONED",
                    "description": flag_note
                }
                inconsistency_flags.append(issue)
                image_positive_omitted.append(issue)
                is_evaluable = True

            # Category 3: Text-Positive / Image-Low-Confidence (Text PRESENT, but DenseNet prob < 0.20)
            elif pred_prob < 0.20 and has_positive:
                status = "MISMATCH_TEXT_POS_IMAGE_LOW_CONF"
                category = "TEXT_POS_IMAGE_LOW_CONF"
                flag_note = f"TEXT-POSITIVE / IMAGE-LOW-CONFIDENCE: BioBERT report analysis extracted PRESENT for '{path_name}' ('{matched_findings[0]['sentence']}'), but DenseNet121 visual probability is low ({pred_prob*100:.1f}%)."
                issue = {
                    "pathology": path_name,
                    "category": category,
                    "issue_type": "Text-Positive / Image-Low-Confidence Finding",
                    "severity": "LOW",
                    "densenet_probability": round(pred_prob, 4),
                    "biobert_report_status": "PRESENT",
                    "description": flag_note
                }
                inconsistency_flags.append(issue)
                text_positive_low_conf.append(issue)
                is_evaluable = True

            # Category 4: Model Uncertainty / Borderline (DenseNet prob 0.40 - 0.69 or Text UNCERTAIN)
            elif (0.40 <= pred_prob < 0.70 and not has_positive) or (pred_prob >= 0.40 and has_uncertain):
                status = "MODEL_UNCERTAINTY_BORDERLINE"
                category = "MODEL_UNCERTAINTY_BORDERLINE"
                rep_st = "UNCERTAIN" if has_uncertain else "UNMENTIONED"
                flag_note = f"MODEL UNCERTAINTY / BORDERLINE: DenseNet121 visual probability for '{path_name}' is in equivocal range ({pred_prob*100:.1f}%), report status is '{rep_st}'."
                issue = {
                    "pathology": path_name,
                    "category": category,
                    "issue_type": "Model Uncertainty / Borderline Prediction",
                    "severity": "INFORMATIONAL",
                    "densenet_probability": round(pred_prob, 4),
                    "biobert_report_status": rep_st,
                    "description": flag_note
                }
                model_uncertainties.append(issue)
                is_evaluable = False

            # Category 5: Concordant / Normal Observation
            elif is_mentioned or has_normal:
                status = "CONCORDANT"
                category = "CONCORDANT_NORMAL"
                concordant_count += 1
                concordant_normals.append({"pathology": path_name, "status": status})
                is_evaluable = True

            else:
                status = "UNMENTIONED"
                category = "CONCORDANT_NORMAL"
                is_evaluable = False

            if is_evaluable:
                evaluable_count += 1

            comparison_matrix.append({
                "pathology": path_name,
                "densenet_prob": round(pred_prob, 4),
                "report_status": "PRESENT" if has_positive else ("ABSENT" if has_negative else ("NORMAL" if has_normal else ("UNCERTAIN" if has_uncertain else "UNMENTIONED"))),
                "category": category,
                "status": status,
                "flag_note": flag_note,
                "is_evaluable": is_evaluable
            })

        total_flags = len(inconsistency_flags)
        if evaluable_count > 0:
            consistency_score = round((concordant_count / evaluable_count) * 100.0, 1)
            status_text = f"{consistency_score}% Concordance ({concordant_count}/{evaluable_count} Pathologies Evaluated)"
        else:
            consistency_score = 100.0
            status_text = "100.0% Concordance (0 Specific Pathologies Mentioned)"

        return {
            "consistency_score": consistency_score,
            "status_text": status_text,
            "inconsistencies_found": len(confirmed_contradictions) > 0 or len(image_positive_omitted) > 0,
            "flag_count": total_flags,
            "evaluable_count": evaluable_count,
            "category_counts": {
                "confirmed_contradictions": len(confirmed_contradictions),
                "image_positive_omitted": len(image_positive_omitted),
                "text_positive_low_conf": len(text_positive_low_conf),
                "model_uncertainty": len(model_uncertainties),
                "concordant_normal": len(concordant_normals)
            },
            "confirmed_contradictions": confirmed_contradictions,
            "image_positive_omitted": image_positive_omitted,
            "text_positive_low_conf": text_positive_low_conf,
            "model_uncertainties": model_uncertainties,
            "flagged_issues": inconsistency_flags,
            "comparison_matrix": comparison_matrix
        }

    def process_full_report(
        self,
        report_text: str,
        pathology_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Master execution pipeline:
          1. Extracts BioBERT embeddings & token representations
          2. Extracts Biomedical NER entities
          3. Extracts positive (PRESENT), negative (ABSENT), and uncertain radiology findings with laterality & severity
          4. Compares with DenseNet121 14-pathology predictions & calculates actual concordance score
          5. Generates structured BioBERT verification package for LLM refinement
        """
        biobert_emb = self.extract_biobert_embeddings(report_text)
        ner_entities = self.extract_ner_entities(report_text)
        structured_findings = self.extract_radiology_findings(report_text)
        comparison_results = self.compare_with_densenet(structured_findings, pathology_predictions or {})

        positives = [f for f in structured_findings if f["status"] == "PRESENT"]
        negatives = [f for f in structured_findings if f["status"] == "ABSENT"]
        uncertains = [f for f in structured_findings if f["status"] == "UNCERTAIN"]

        return {
            "report_text": report_text,
            "biobert_embeddings": biobert_emb,
            "ner_entities": ner_entities,
            "structured_findings": structured_findings,
            "summary": {
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "uncertain_count": len(uncertains),
                "positives": positives,
                "negatives": negatives,
                "uncertains": uncertains
            },
            "densenet_verification": comparison_results
        }


_global_biobert_engine = None

def get_biobert_engine() -> BioBERTBiomedicalNLPEngine:
    """Returns a singleton BioBERT Biomedical NLP engine instance."""
    global _global_biobert_engine
    if _global_biobert_engine is None:
        _global_biobert_engine = BioBERTBiomedicalNLPEngine()
    return _global_biobert_engine
