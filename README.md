# Medical Image Report Generation

An automated medical image analysis and radiology report generation pipeline using Python, PyTorch, and MIMIC-CXR chest X-ray dataset processing scripts.

## Project Overview

This repository contains scripts for:
- MIMIC-CXR dataset verification and analysis
- Chest X-ray image mapping and study matching
- Data preprocessing and processed dataset generation
- Model training & evaluation pipelines for medical image report generation

## Repository Structure

```
.
├── 01_dataset_check.py           # Inspect dataset schema and missing values
├── 02_analyze_dataset.py         # Detailed statistical analysis of radiology reports
├── 03_check_studies.py           # Match patient study IDs with images
├── 04_check_mapping.py           # Verify DICOM/JPEG image path mappings
├── 05_create_processed_dataset.py# Preprocess and clean radiology text and image paths
├── 06_check_processed.py         # Quick sanity checks on processed dataset
├── 07_test_xray.py              # Test script for single image inspection
├── project_code.py               # Combined baseline model and training script
└── code snippets.txt             # Reference snippets and notes
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Medical_Image_Report.git
   cd Medical_Image_Report
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy torch torchvision pillow
   ```

4. **Data Prerequisites:**
   Download the MIMIC-CXR augmented dataset (`mimic_cxr_aug_train.csv`) and place it in the root directory.

## License

MIT License
