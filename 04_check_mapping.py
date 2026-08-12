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
print("CHECKING IMAGE / STUDY MAPPING")
print("=" * 70)

for i in range(5):

    row = df.iloc[i]

    subject_id = row["subject_id"]

    images = parse_list(row["image"])
    ap_images = parse_list(row["AP"])
    pa_images = parse_list(row["PA"])
    lateral_images = parse_list(row["Lateral"])
    views = parse_list(row["view"])
    reports = parse_list(row["text"])

    print("\n" + "=" * 70)
    print("PATIENT:", subject_id)
    print("=" * 70)

    print("Total images :", len(images))
    print("AP images    :", len(ap_images))
    print("PA images    :", len(pa_images))
    print("Lateral     :", len(lateral_images))
    print("Views       :", len(views))
    print("Reports     :", len(reports))

    print("\n--- AP IMAGES ---")

    for image in ap_images:
        print(image)

    print("\n--- PA IMAGES ---")

    for image in pa_images:
        print(image)

    print("\n--- LATERAL IMAGES ---")

    for image in lateral_images:
        print(image)

    print("\n--- REPORTS ---")

    for j, report in enumerate(reports):

        print("\nREPORT", j)
        print(report[:300])

print("\n" + "=" * 70)
print("MAPPING CHECK COMPLETED")
print("=" * 70)