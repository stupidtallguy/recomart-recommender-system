import pandas as pd

df = pd.read_parquet(
    "data/processed/interactions.parquet"
)

print(df.shape)

print(df["user_id"].nunique())

print(df["order_number"].max())

print(df.head())