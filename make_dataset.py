"""
Builds a small messy customer dataset to practise preprocessing on.
No internet needed - everything is generated with a fixed seed so the
results are the same every time you run it.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

cities = ["Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Salem", "Coimbatore"]
edu = ["High School", "Bachelors", "Masters", "PhD"]
member = ["Basic", "Silver", "Gold"]

age = rng.normal(38, 11, n).clip(18, 70).round()
income = (age * 900 + rng.normal(25000, 12000, n)).clip(8000, None).round(-2)
tenure = rng.uniform(0, 10, n).round(1)
score = (50 + (income - income.mean()) / income.std() * 8 + rng.normal(0, 18, n)).clip(1, 100).round()
gold_prob = np.where(income > np.percentile(income, 70), 0.5, 0.15)
membership = np.where(rng.random(n) < gold_prob, "Gold", rng.choice(["Basic", "Silver"], n))
churn_p = 0.15 + 0.25 * (score < 35) - 0.1 * (tenure > 5)
churned = np.where(rng.random(n) < churn_p, "Yes", "No")

df = pd.DataFrame({
    "customer_id": np.arange(1001, 1001 + n),
    "age": age,
    "gender": rng.choice(["Male", "Female"], n),
    "city": rng.choice(cities, n),
    "education": rng.choice(edu, n, p=[0.25, 0.4, 0.3, 0.05]),
    "annual_income": income,
    "spending_score": score,
    "tenure_years": tenure,
    "signup_date": pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 3300, n), unit="D"),
    "membership": membership,
    "country": "India",
    "churned": churned,
})

# ---- now make it dirty ----
# missing values
for col, frac in [("age", 0.06), ("annual_income", 0.08), ("education", 0.05),
                  ("spending_score", 0.04), ("city", 0.03)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan

# inconsistent text
g_map = {"Male": ["Male", "male", "M", "MALE "], "Female": ["Female", "female", "F", " Female"]}
df["gender"] = df["gender"].apply(lambda x: rng.choice(g_map[x]))
df["city"] = df["city"].apply(lambda x: x if pd.isna(x) or rng.random() > 0.15 else x.lower())

# a few impossible / extreme values
df.loc[[12, 87, 301], "age"] = [250, -5, 140]
df.loc[[45, 210], "annual_income"] = [2_400_000, 3_100_000]
df.loc[130, "tenure_years"] = -2.0

# duplicate rows
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)

# dates stored as text, two formats
df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
mask = rng.random(len(df)) < 0.2
df.loc[mask, "signup_date"] = pd.to_datetime(df.loc[mask, "signup_date"]).dt.strftime("%d/%m/%Y")

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("data/raw_customers.csv", index=False)
print("saved", df.shape)
