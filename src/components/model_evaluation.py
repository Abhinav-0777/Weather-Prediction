import pandas as pd

df = pd.read_csv("weatherAUS.csv")

print(df['WindGustDir'].nunique())
print(df['WindDir9am'].nunique())
print(df['WindDir3pm'].nunique())
print(df['Location'].nunique())
print(df['RainToday'].nunique())