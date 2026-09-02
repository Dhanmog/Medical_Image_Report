import pandas as pd

df = pd.read_csv("processed_dataset.csv")

print("=" * 60)
print("PROCESSED DATASET")
print("=" * 60)

print("\nNumber of rows:")
print(len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nUnique patients:")
print(df["subject_id"].nunique())

print("\nUnique studies:")
print(df["study_id"].nunique())

print("\nMissing values:")
print(df.isnull().sum())