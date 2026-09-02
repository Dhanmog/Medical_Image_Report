import os
import pandas as pd

# Define paths for new dataset
METADATA_PATH = "shenzhen_metadata.csv"
IMAGE_DIR = os.path.join("images", "images")
OUTPUT_CSV = "shenzhen_processed.csv"

def prepare_shenzhen_dataset():
    print("=" * 60)
    print("PREPROCESSING SHENZHEN CHEST X-RAY DATASET")
    print("=" * 60)
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
        
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata with {len(df)} rows.")
    print("Columns:", df.columns.tolist())
    
    processed_rows = []
    missing_images = 0
    
    for idx, row in df.iterrows():
        filename = str(row["study_id"]).strip()
        image_path = os.path.join(IMAGE_DIR, filename)
        
        # Verify image existence
        if not os.path.exists(image_path):
            missing_images += 1
            continue
            
        findings = str(row["findings"]).strip()
        sex = str(row["sex"]).strip() if pd.notnull(row["sex"]) else "Unknown"
        age = str(row["age"]).strip() if pd.notnull(row["age"]) else "Unknown"
        
        # Formulate report string from clinical findings
        if findings.lower() == "normal":
            report = "chest x-ray shows normal lungs and heart."
        else:
            report = f"chest x-ray findings indicate {findings}."
            
        processed_rows.append({
            "study_id": filename,
            "image_path": image_path,
            "sex": sex,
            "age": age,
            "findings": findings,
            "report": report
        })
        
    processed_df = pd.DataFrame(processed_rows)
    print(f"\nSuccessfully validated {len(processed_df)} images (Missing: {missing_images}).")
    
    processed_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved processed Shenzhen dataset to: {OUTPUT_CSV}")
    print("\nSample rows:")
    print(processed_df.head())
    print("=" * 60)

if __name__ == "__main__":
    prepare_shenzhen_dataset()
