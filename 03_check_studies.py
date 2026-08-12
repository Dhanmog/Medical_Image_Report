import pandas as pd
import ast

CSV_PATH = "mimic_cxr_aug_train.csv"

df = pd.read_csv(CSV_PATH)


def parse_list(value):
    try:
        return ast.literal_eval(value)
    except:
        return []


print("=" * 70)
print("STUDY-IMAGE-REPORT ANALYSIS")
print("=" * 70)

for i in range(10):

    row = df.iloc[i]

    subject_id = row["subject_id"]

    images = parse_list(row["image"])
    reports = parse_list(row["text"])
    views = parse_list(row["view"])

    print("\n" + "=" * 70)
    print("PATIENT:", subject_id)
    print("=" * 70)

    print("Images :", len(images))
    print("Views  :", len(views))
    print("Reports:", len(reports))

    study_ids = []

    for image in images:

        parts = image.split("/")

        if len(parts) >= 4:
            study_id = parts[3]
            study_ids.append(study_id)

    unique_studies = list(dict.fromkeys(study_ids))

    print("\nStudies found:", len(unique_studies))

    for study in unique_studies:

        count = study_ids.count(study)

        print("   ", study, "->", count, "images")

    print("\nReports:")

    for j, report in enumerate(reports):

        print(
            f"Report {j}:",
            report[:200].replace("\n", " ")
        )


print("\n" + "=" * 70)
print("CHECK COMPLETED")
print("=" * 70)