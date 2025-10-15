import pandas as pd
import os

# Read the CSV file
csv_path = './secondary-cvd-risk.csv'
parquet_path = './secondary-cvd-risk.parquet'

# Convert CSV to Parquet
df = pd.read_csv(csv_path, encoding='ascii')
df.to_parquet(parquet_path, index=False)

print(f"Successfully converted {csv_path} to {parquet_path}")
