import pandas as pd

input_path =r"Data\options_2020.csv" # <- change this to your actual path


df = pd.read_csv(input_path)
print(df.columns)
print(df)
print(df["impl_volatility"])
print(len(df))