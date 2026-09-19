from sklearn.datasets import fetch_openml

df = fetch_openml(name="adult", version=2, as_frame=True).frame
df.to_csv("adult_income.csv", index=False)
print(df.shape)
print(df["class"].value_counts())