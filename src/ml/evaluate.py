import pandas as pd

df = pd.read_csv("data/processed/featured_dataset.csv")

print(df.select_dtypes(include="object").columns.tolist())