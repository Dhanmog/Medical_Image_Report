"""
================================================================================
UNIFESP REAL DICOM ANATOMICAL BODY PART CLASSIFIER & CHEST VALIDATION MODULE
================================================================================
This module serves as a mandatory anatomical safety gate before any chest X-ray
disease classification (DenseNet121) or radiology report generation model is executed.

Model Artifact : models/chest_validator/unifesp_mobilenet_v2_real.keras
Class Mapping  : models/chest_validator/unifesp_class_mapping.json
Default Gate   : 0.60 (60% Chest Confidence, Validation-Calibrated Experimental Threshold)

Key Safeguards:
  1. DICOM & Standard Image Support: pydicom handling, MONOCHROME1 inversion, CLAHE.
  2. Multi-Label Classification & Ambiguity Protection: Uncertainty & margin checks.
  3. Safe Failure Handling: Invalid files, decoding errors, or missing models trigger st.stop().
================================================================================
"""

import os
import io
import cv2
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

try:
    import pydicom
except ImportError:
    pydicom = None

# Validation-calibrated experimental threshold (0.60 / 60%)
DEFAULT_CHEST_THRESHOLD = 0.60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_REAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "chest_validator", "unifesp_mobilenet_v2_real.keras")
SAVED_FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "models", "chest_validator", "chest_validator_mobilenet.keras")
MAPPING_JSON_PATH = os.path.join(BASE_DIR, "models", "chest_validator", "unifesp_class_mapping.json")

# Standard 22 UNIFESP Body Part Categories
UNIFESP_22_CLASSES = [
    "Abdomen",        # 0
    "Ankle",          # 1
    "Cervical Spine", # 2
    "Chest",          # 3
    "Clavicle",       # 4
    "Elbow",          # 5
    "Feet",           # 6
    "Finger",         # 7
    "Forearm",        # 8
    "Hand",           # 9
    "Hip",            # 10
    "Knee",           # 11
    "Lower Leg",      # 12
    "Lumbar Spine",   # 13
    "Other",          # 14
    "Pelvis",         # 15
    "Radius",         # 16
    "Shoulder",       # 17
    "Skull",          # 18
    "Thigh",          # 19
    "Thoracic Spine", # 20
    "Wrist"           # 21
]

_VALIDATOR_MODEL = None
_CLASS_MAPPING = None


def load_class_mapping():
    """Loads UNIFESP class mapping dictionary from JSON if available."""
    global _CLASS_MAPPING
    if _CLASS_MAPPING is not None:
        return _CLASS_MAPPING

    if os.path.exists(MAPPING_JSON_PATH):
        try:
            with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                int_mapping = {int(k): v for k, v in mapping.items()}
                # Verify mapping contains exactly 22 classes and index 3 maps to Chest
                if len(int_mapping) == 22 and int_mapping.get(3, "").lower() == "chest":
                    _CLASS_MAPPING = int_mapping
                    return _CLASS_MAPPING
        except Exception as e:
            print(f"Notice: Loading class mapping from {MAPPING_JSON_PATH} failed: {e}")

    _CLASS_MAPPING = {i: name for i, name in enumerate(UNIFESP_22_CLASSES)}
    return _CLASS_MAPPING


def load_validator_model():
    """
    Initializes and caches the trained UNIFESP MobileNetV2 body-part classification model.
    Loads fine-tuned weights from models/chest_validator/unifesp_mobilenet_v2_real.keras if present.
    """
    global _VALIDATOR_MODEL
    if _VALIDATOR_MODEL is not None:
        return _VALIDATOR_MODEL

    # 1. Primary Model Path (Trained on Real DICOM UNIFESP Dataset)
    if os.path.exists(SAVED_REAL_MODEL_PATH):
        try:
            _VALIDATOR_MODEL = keras.models.load_model(SAVED_REAL_MODEL_PATH, compile=False)
            return _VALIDATOR_MODEL
        except Exception as e:
            print(f"Notice: Loading real UNIFESP model failed ({e}), checking fallback path...")

    # 2. Fallback Model Path
    if os.path.exists(SAVED_FALLBACK_MODEL_PATH):
        try:
            _VALIDATOR_MODEL = keras.models.load_model(SAVED_FALLBACK_MODEL_PATH, compile=False)
            return _VALIDATOR_MODEL
        except Exception:
            pass

    print("Warning: Could not load trained UNIFESP validator model.")
    return None


def preprocess_image_bytes_for_classifier(image_bytes, target_size=(224, 224)):
    """
    Exact training preprocessing logic from scripts/train_unifesp_real_dataset.py:
      - Explicitly detects DICOM input (header signature b'DICM' or pydicom check).
      - If DICOM parse or pixel decoding fails, raises a clear ValueError without falling back to OpenCV/PIL.
      - Standard images (JPG/PNG/JPEG) use cv2.imdecode / PIL.
      - Applies DICOM RescaleSlope / RescaleIntercept & MONOCHROME1 inversion.
      - Applies CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=(8,8)).
      - Resizes to 224x224 and normalizes float32 [0.0, 1.0] for classifier.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")

    is_dicom = False
    img_gray = None

    # Detect DICOM input explicitly
    is_dicom_file = False
    if len(image_bytes) >= 132 and image_bytes[128:132] == b'DICM':
        is_dicom_file = True
    elif pydicom is not None:
        try:
            is_dicom_file = pydicom.misc.is_dicom(io.BytesIO(image_bytes))
        except Exception:
            pass

    if is_dicom_file:
        if pydicom is None:
            raise ValueError("DICOM image detected, but pydicom library is not installed.")
        try:
            ds = pydicom.dcmread(io.BytesIO(image_bytes))
            img_raw = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            if slope != 1.0 or intercept != 0.0:
                img_raw = img_raw * slope + intercept

            photo = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')).strip().upper()
            if photo == 'MONOCHROME1':
                img_raw = np.max(img_raw) - img_raw

            min_val, max_val = np.min(img_raw), np.max(img_raw)
            if max_val > min_val:
                img_gray = ((img_raw - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                img_gray = np.zeros(img_raw.shape, dtype=np.uint8)
            is_dicom = True
        except Exception as e:
            raise ValueError(f"DICOM parsing or pixel decoding failed: {e}")
    else:
        # Standard Image decoding (JPG/PNG/JPEG)
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
                img_gray = np.array(pil_img)
            except Exception:
                raise ValueError("Could not decode standard image format (JPG/PNG).")

    # Apply CLAHE contrast enhancement
    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)

    # Normalize float32 range [0.0, 1.0] for classifier
    img_norm = img_clahe.astype(np.float32) / 255.0
    img_norm_3ch = np.stack([img_norm, img_norm, img_norm], axis=-1)

    # Convert display RGB uint8 array for Streamlit preview
    img_display_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    return img_norm_3ch, img_display_rgb, is_dicom


def classify_body_part(image_bytes, uncertainty_margin: float = 0.12):
    """
    Classifies the uploaded X-ray image using the UNIFESP body-part dataset taxonomy.
    Uses dynamically loaded class mapping from unifesp_class_mapping.json.

    Returns:
      predicted_label (str): Top predicted UNIFESP body part ('Chest', 'Abdomen', 'Skull', etc.)
      chest_confidence (float): Confidence score for Chest category [0.0, 1.0]
      all_probs (dict): Full probability mapping across all UNIFESP classes
      is_uncertain (bool): True if prediction margin is small or top score is low
      uncertainty_details (dict): Metadata regarding prediction certainty
      display_rgb (np.ndarray): Display RGB image array for Streamlit
    """
    try:
        img_norm_3ch, display_rgb, is_dicom = preprocess_image_bytes_for_classifier(image_bytes)

        model = load_validator_model()
        mapping = load_class_mapping()  # {0: "Abdomen", 1: "Ankle", ..., 3: "Chest", ...}

        if model is None:
            err_details = {
                "is_uncertain": True,
                "uncertainty_reason": "Validator model file missing or failed to initialize",
                "top1_prob": 0.0,
                "margin": 0.0,
                "is_dicom": is_dicom
            }
            all_probs_err = {name: 0.0 for name in mapping.values()}
            return "Other", 0.0, all_probs_err, True, err_details, display_rgb

        inp = np.expand_dims(img_norm_3ch, axis=0)
        preds = model.predict(inp, verbose=0)[0]
        num_outputs = len(preds)

        if num_outputs == len(mapping):
            probs_dict = {mapping[i]: float(preds[i]) for i in range(num_outputs)}
        elif num_outputs == 8:
            legacy_8 = ["CHEST", "ABDOMEN", "SKULL", "SPINE", "UPPER_LIMB", "LOWER_LIMB", "PELVIS", "OTHER"]
            probs_dict = {name: 0.0 for name in mapping.values()}
            c_map = {"CHEST": "Chest", "ABDOMEN": "Abdomen", "SKULL": "Skull", "SPINE": "Lumbar Spine",
                     "UPPER_LIMB": "Hand", "LOWER_LIMB": "Knee", "PELVIS": "Pelvis", "OTHER": "Other"}
            for i in range(8):
                target_cls = c_map.get(legacy_8[i], "Other")
                if target_cls in probs_dict:
                    probs_dict[target_cls] = float(preds[i])
        else:
            probs_dict = {mapping.get(i, f"Class_{i}"): float(preds[i]) for i in range(min(num_outputs, len(mapping)))}

        chest_confidence = float(probs_dict.get("Chest", 0.0))

        # Sort probabilities to evaluate margin & uncertainty
        sorted_probs = sorted(probs_dict.values(), reverse=True)
        top1_prob = sorted_probs[0]
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top1_prob - top2_prob

        pred_class = max(probs_dict, key=probs_dict.get)

        # Check top two classes to handle Chest vs Thoracic Spine lateral view synergy
        sorted_classes = sorted(probs_dict.keys(), key=lambda k: probs_dict[k], reverse=True)
        top1_class = sorted_classes[0]
        top2_class = sorted_classes[1] if len(sorted_classes) > 1 else None

        # Both classes belong to the thoracic cage / chest anatomy (frontal vs lateral views)
        thoracic_set = {"Chest", "Thoracic Spine"}
        both_thoracic = (top1_class in thoracic_set) and (top2_class in thoracic_set)
        thoracic_combined = float(probs_dict.get("Chest", 0.0) + probs_dict.get("Thoracic Spine", 0.0))

        # Uncertainty Evaluation Criteria
        is_uncertain = False
        uncertainty_reason = None

        if top1_prob < 0.40 and not (both_thoracic and thoracic_combined >= 0.55):
            is_uncertain = True
            uncertainty_reason = f"Low top-1 confidence ({top1_prob*100:.1f}% < 40%)"
        elif margin < uncertainty_margin and top1_prob < 0.70 and not both_thoracic:
            is_uncertain = True
            uncertainty_reason = f"Narrow prediction margin ({margin:.3f} < {uncertainty_margin})"

        uncertainty_details = {
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "top1_prob": round(top1_prob, 4),
            "top2_prob": round(top2_prob, 4),
            "margin": round(margin, 4),
            "is_dicom": is_dicom
        }

        all_probs = {k: round(v, 4) for k, v in probs_dict.items()}
        return pred_class, chest_confidence, all_probs, is_uncertain, uncertainty_details, display_rgb

    except Exception as e:
        print(f"Error in classify_body_part: {e}")
        mapping = load_class_mapping()
        err_details = {
            "is_uncertain": True,
            "uncertainty_reason": f"Image processing failure: {str(e)}",
            "top1_prob": 0.0,
            "top2_prob": 0.0,
            "margin": 0.0,
            "is_dicom": False
        }
        all_probs_err = {name: 0.0 for name in mapping.values()}
        if "Other" in all_probs_err:
            all_probs_err["Other"] = 1.0
        return "Other", 0.0, all_probs_err, True, err_details, None


def validate_chest_xray(image_bytes: bytes, threshold: float = DEFAULT_CHEST_THRESHOLD):
    """
    Main entry point for Chest X-Ray Validation and UNIFESP Body Part Classification.

    Args:
        image_bytes (bytes): Raw bytes of uploaded image
        threshold (float): Confidence threshold for Chest validation (default 0.60, validation-calibrated)

    Returns:
        is_chest (bool): True if image is confidently Chest, confidence >= threshold, AND NOT uncertain
        confidence (float): Chest confidence score between 0.0 and 1.0
        detected_label (str): Detected UNIFESP anatomical category
        details (dict): Full metadata, probability breakdown, uncertainty details, and display array
    """
    if not image_bytes:
        return False, 0.0, "Other", {
            "error": "Empty image bytes",
            "is_chest": False,
            "is_uncertain": True,
            "uncertainty_reason": "Empty image upload",
            "detected_label": "Other",
            "confidence": 0.0
        }

    detected_label, confidence, all_probs, is_uncertain, unc_details, display_rgb = classify_body_part(
        image_bytes, uncertainty_margin=0.12
    )

    p_chest = float(all_probs.get("Chest", 0.0))
    p_thoracic_spine = float(all_probs.get("Thoracic Spine", 0.0))
    thoracic_score = p_chest + p_thoracic_spine
    det_lower = detected_label.lower()

    # Determine view type: Frontal vs Lateral
    if det_lower == "thoracic spine" or (p_thoracic_spine > 0.35 and p_thoracic_spine >= p_chest):
        view_type = "Lateral"
        canonical_label = "Chest (Lateral View)"
    else:
        view_type = "Frontal"
        canonical_label = "Chest (Frontal View)"

    # Check if the radiograph belongs to thoracic anatomy
    is_thoracic = det_lower in {"chest", "thoracic spine"}

    # Validation criteria supporting both Frontal (PA/AP) and Lateral Chest X-rays:
    # 1. Detected anatomical category is Chest or Thoracic Spine (Lateral projection)
    # 2. Thoracic confidence (Chest or Chest + Thoracic Spine) meets the calibrated threshold
    # 3. Prediction is not genuinely uncertain (competing with non-chest organs)
    is_chest = bool(
        is_thoracic and
        (thoracic_score >= threshold or p_chest >= threshold or confidence >= threshold) and
        (not is_uncertain)
    )

    effective_confidence = max(confidence, thoracic_score)

    details = {
        "is_chest": is_chest,
        "is_uncertain": is_uncertain,
        "uncertainty_reason": unc_details.get("uncertainty_reason"),
        "confidence": effective_confidence,
        "confidence_percentage": round(effective_confidence * 100.0, 1),
        "threshold": threshold,
        "threshold_percentage": round(threshold * 100.0, 1),
        "detected_label": canonical_label if is_chest else detected_label,
        "raw_detected_label": detected_label,
        "view_type": view_type if is_chest else "Non-Chest",
        "unifesp_category": detected_label,
        "all_probabilities": all_probs,
        "margin": unc_details.get("margin", 0.0),
        "top1_prob": unc_details.get("top1_prob", 0.0),
        "is_dicom": unc_details.get("is_dicom", False),
        "display_image": display_rgb
    }

    return is_chest, effective_confidence, (canonical_label if is_chest else detected_label), details
