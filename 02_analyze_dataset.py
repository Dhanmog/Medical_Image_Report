import pandas as pd
import ast

CSV_PATH = "mimic_cxr_aug_train.csv"

df = pd.read_csv(CSV_PATH)


def parse_list(value):
    try:
        return ast.literal_eval(value)
    except:
        return []


print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print("Total rows:", len(df))

# Count images and reports in each row
image_lengths = df["image"].apply(lambda x: len(parse_list(x)))
report_lengths = df["text"].apply(lambda x: len(parse_list(x)))

print("\nImage count statistics:")
print(image_lengths.describe())

print("\nReport count statistics:")
print(report_lengths.describe())

# Check whether image and report counts are equal
matching = image_lengths == report_lengths

print("\nRows where image count = report count:")
print(matching.sum())

print("\nRows where image count != report count:")
print((~matching).sum())

# Show first 5 rows in detail
for i in range(5):

    print("\n" + "=" * 60)
    print("ROW:", i)
    print("=" * 60)

    images = parse_list(df.iloc[i]["image"])
    views = parse_list(df.iloc[i]["view"])
    reports = parse_list(df.iloc[i]["text"])

    print("Subject ID:", df.iloc[i]["subject_id"])

    print("\nNumber of images:", len(images))
    print("Number of views:", len(views))
    print("Number of reports:", len(reports))

    print("\nImages:")

    for j, image in enumerate(images[:10]):
        print(j, image)

    print("\nViews:")

    for j, view in enumerate(views[:10]):
        print(j, view)

    print("\nReports:")

    for j, report in enumerate(reports[:5]):
        print("\nREPORT", j)
        print(report[:500])

print("\nDataset analysis completed.")