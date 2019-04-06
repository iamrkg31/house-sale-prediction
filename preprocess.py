import pandas as pd

df = pd.read_csv("data/data.csv")

print(df.columns)



df["date"] = df["date"].replace(to_replace="(2014|2015)(.*)", value='\\1', regex=True)
df["date"] = df["date"].replace({"2014":0, "2015":1})


print(df["yr_renovated"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

sns.jointplot(x="grade", y="price", data=df, kind = 'reg', size = 7)

# Create 2 new columns for the analysis
df['sqft_basement2'] = df['sqft_basement'].apply(lambda x: x if x > 0 else None)
df['yr_renovated2'] = df['yr_renovated'].apply(lambda x: x if x > 0 else None)

# Show the new plots with paerson correlation
sns.jointplot(x="sqft_basement2", y="price", data=df, kind = 'reg', dropna=True, size = 5)
sns.jointplot(x="yr_renovated2", y="price", data=df, kind = 'reg', dropna=True, size = 5)
plt.show()