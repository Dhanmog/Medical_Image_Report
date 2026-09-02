import pandas as pd
import ast
import os

CSV_PATH = "mimic_cxr_aug_train.csv"
OUTPUT_PATH = "processed_dataset.csv"


def parse_list(value):
    try:
        return ast.literal_eval(value)
    except:
        return []


# Load original dataset
df = pd.read_csv(CSV_PATH)

processed_rows = []

print("Processing dataset...")

for index, row in df.iterrows():

    subject_id = row["subject_id"]

    images = parse_list(row["image"])
    views = parse_list(row["view"])
    reports = parse_list(row["text"])

    # We only process rows where there is at least
    # one image and one report
    if len(images) == 0 or len(reports) == 0:
        continue

    # Get study IDs from image paths
    study_images = {}

    for image_path in images:

        parts = image_path.split("/")

        if len(parts) >= 4:

            study_id = parts[3]

            if study_id not in study_images:
                study_images[study_id] = []

            study_images[study_id].append(image_path)

    # IMPORTANT:
    # At this stage we only create study/image information.
    # We do NOT blindly assign reports to studies.
    #
    # For now, use the first report as a temporary report.
    # We will improve this mapping after checking the actual
    # dataset structure.

    first_report = reports[0]

    for study_id, study_image_list in study_images.items():

        for image_path in study_image_list:

            processed_rows.append({
                "subject_id": subject_id,
                "study_id": study_id,
                "image_path": image_path,
                "report": first_report
            })


# Create DataFrame
processed_df = pd.DataFrame(processed_rows)

print("\nProcessing completed.")

print("Total processed images:", len(processed_df))

print("\nColumns:")
print(processed_df.columns.tolist())

print("\nFirst 5 rows:")
print(processed_df.head())

# Save
processed_df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved as:")
print(OUTPUT_PATH)