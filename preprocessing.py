"""
Preprocessing pipeline for the customer dataset.
Run:  python preprocessing.py
Input : data/raw_customers.csv
Output: data/clean_customers.csv (encoded, not scaled)
        data/clean_customers_scaled.csv (ready for ML)
        plots/*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)


def step(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. load + first look
# ---------------------------------------------------------------
step("STEP 1: Load data")
df = pd.read_csv("data/raw_customers.csv")
print("shape:", df.shape)
print(df.dtypes)
print("\nmissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nsummary stats:")
print(df.describe().round(2))

# keep a copy for before/after plots
raw = df.copy()

# ---------------------------------------------------------------
# 2. duplicates
# ---------------------------------------------------------------
step("STEP 2: Duplicate rows")
dups = df.duplicated().sum()
print("duplicate rows found:", dups)
df = df.drop_duplicates().reset_index(drop=True)
print("shape after dropping:", df.shape)

# ---------------------------------------------------------------
# 3. messy text (spaces, case, short forms)
# ---------------------------------------------------------------
step("STEP 3: Clean text columns")
print("gender values before:", sorted(df["gender"].unique()))
df["gender"] = df["gender"].str.strip().str.lower().str[0].map({"m": "Male", "f": "Female"})
print("gender values after :", sorted(df["gender"].unique()))

print("city values before  :", sorted(df["city"].dropna().unique()))
df["city"] = df["city"].str.strip().str.title()
print("city values after   :", sorted(df["city"].dropna().unique()))

# ---------------------------------------------------------------
# 4. dates
# ---------------------------------------------------------------
step("STEP 4: Fix date column")
# two formats are mixed (YYYY-MM-DD and DD/MM/YYYY), so parse each separately
iso = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")
dmy = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")
df["signup_date"] = iso.fillna(dmy)
print("unparsed dates:", df["signup_date"].isnull().sum())
df["signup_year"] = df["signup_date"].dt.year
print(df["signup_year"].value_counts().sort_index().to_dict())

# some duplicates only differed in date format, so check again now that dates match
dups2 = df.duplicated().sum()
print("hidden duplicates found after fixing dates:", dups2)
df = df.drop_duplicates().reset_index(drop=True)
print("shape now:", df.shape)

# ---------------------------------------------------------------
# 5. impossible values -> treat as missing
# ---------------------------------------------------------------
step("STEP 5: Impossible values")
bad_age = ((df["age"] < 18) | (df["age"] > 100)).sum()
bad_tenure = (df["tenure_years"] < 0).sum()
print("ages outside 18-100:", bad_age, "| negative tenure:", bad_tenure)
print("bad ages:", df.loc[(df["age"] < 18) | (df["age"] > 100), "age"].tolist())
df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
df.loc[df["tenure_years"] < 0, "tenure_years"] = np.nan

# ---------------------------------------------------------------
# 6. missing values
# ---------------------------------------------------------------
step("STEP 6: Missing values")
print("before:\n", df.isnull().sum()[df.isnull().sum() > 0])

# numeric -> median (robust to outliers like the 3,100,000 income)
for col in ["age", "annual_income", "spending_score", "tenure_years"]:
    med = df[col].median()
    df[col] = df[col].fillna(med)
    print(f"{col}: filled with median = {med}")

# categorical -> most common value
for col in ["education", "city"]:
    mode = df[col].mode()[0]
    df[col] = df[col].fillna(mode)
    print(f"{col}: filled with mode = {mode}")

print("\nmissing after:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------
# 7. outliers (IQR)
# ---------------------------------------------------------------
step("STEP 7: Outliers in annual_income")
q1, q3 = np.percentile(df["annual_income"], [25, 75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((df["annual_income"] < low) | (df["annual_income"] > high)).sum()
print(f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
print(f"allowed range: {low:.0f} to {high:.0f}  -> {n_out} outliers")
df["annual_income"] = df["annual_income"].clip(low, high)   # cap, don't delete rows
print("max income after capping:", df["annual_income"].max())

# ---------------------------------------------------------------
# 8. feature selection
# ---------------------------------------------------------------
step("STEP 8: Drop columns we don't need")
print("unique values in country:", df["country"].unique(), "-> constant, no information")
df = df.drop(columns=["country", "customer_id", "signup_date"])
print("dropped: country, customer_id, signup_date (replaced by signup_year)")

# ---------------------------------------------------------------
# 9. encoding
# ---------------------------------------------------------------
step("STEP 9: Encode categorical variables")
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["churned"] = df["churned"].map({"No": 0, "Yes": 1})
# these two have a natural order so ordinal encoding makes sense
df["education"] = df["education"].map({"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3})
df["membership"] = df["membership"].map({"Basic": 0, "Silver": 1, "Gold": 2})
# city has no order -> one-hot
df = pd.get_dummies(df, columns=["city"], prefix="city", dtype=int)
print("columns now:", list(df.columns))
df.to_csv("data/clean_customers.csv", index=False)

# ---------------------------------------------------------------
# 10. scaling
# ---------------------------------------------------------------
step("STEP 10: Normalization / scaling")
num_cols = ["age", "annual_income", "spending_score", "tenure_years", "signup_year"]
scaled = df.copy()
# z-score done by hand with numpy so it's clear what's happening
scaled[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
print(scaled[num_cols].describe().loc[["mean", "std"]].round(3))

# min-max version of income just to compare
inc = df["annual_income"]
inc_minmax = (inc - inc.min()) / (inc.max() - inc.min())
print("\nmin-max income range:", inc_minmax.min(), "to", inc_minmax.max())
scaled.to_csv("data/clean_customers_scaled.csv", index=False)

# ---------------------------------------------------------------
# 11. EDA
# ---------------------------------------------------------------
step("STEP 11: Exploratory data analysis")
print("churn rate:", round(df["churned"].mean() * 100, 1), "%")
print("\nchurn rate by membership:")
print(df.groupby("membership")["churned"].mean().round(3).rename({0: "Basic", 1: "Silver", 2: "Gold"}))

corr = df.drop(columns=[c for c in df.columns if c.startswith("city_")]).corr()
print("\ncorrelation with churned:")
print(corr["churned"].drop("churned").sort_values().round(3))

# plot 1: income before vs after
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(raw["annual_income"].dropna(), bins=40, color="#c0504d")
ax[0].set_title("annual_income - raw")
ax[1].hist(df["annual_income"], bins=40, color="#4f81bd")
ax[1].set_title("annual_income - cleaned")
plt.tight_layout()
plt.savefig("plots/income_before_after.png", dpi=120)
plt.close()

# plot 2: age distribution
plt.figure(figsize=(6, 4))
plt.hist(df["age"], bins=25, color="#4f81bd", edgecolor="white")
plt.title("Age distribution (cleaned)")
plt.xlabel("age")
plt.tight_layout()
plt.savefig("plots/age_hist.png", dpi=120)
plt.close()

# plot 3: correlation heatmap
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("plots/correlation.png", dpi=120)
plt.close()

# plot 4: churn by membership
churn_m = df.groupby("membership")["churned"].mean() * 100
plt.figure(figsize=(5, 4))
plt.bar(["Basic", "Silver", "Gold"], churn_m.values, color=["#9bbb59", "#8064a2", "#f79646"])
plt.ylabel("churn %")
plt.title("Churn rate by membership")
plt.tight_layout()
plt.savefig("plots/churn_by_membership.png", dpi=120)
plt.close()

step("DONE")
print("final shape:", df.shape)
print("saved: data/clean_customers.csv, data/clean_customers_scaled.csv, plots/")
