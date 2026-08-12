import pandas as pd
import ast

CSV_PATH = "mimic_cxr_aug_train.csv"

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst row:")
print(df.iloc[0])


def list_length(value):
    try:
        return len(ast.literal_eval(value))
    except:
        return 0


print("\n==============================")
print("IMAGE INFORMATION")
print("==============================")

image_counts = df["image"].apply(list_length)

print("Minimum images per row:", image_counts.min())
print("Maximum images per row:", image_counts.max())
print("Average images per row:", image_counts.mean())

print("\n==============================")
print("REPORT INFORMATION")
print("==============================")

report_counts = df["text"].apply(list_length)

print("Minimum reports per row:", report_counts.min())
print("Maximum reports per row:", report_counts.max())
print("Average reports per row:", report_counts.mean())

print("\n==============================")
print("FIRST IMAGE PATH")
print("==============================")

first_images = ast.literal_eval(df.iloc[0]["image"])

print(first_images[0])

print("\n==============================")
print("FIRST REPORT")
print("==============================")

first_reports = ast.literal_eval(df.iloc[0]["text"])

print(first_reports[0])

print("\nDataset check completed.")