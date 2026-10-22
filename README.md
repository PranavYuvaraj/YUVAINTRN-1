# Data Preprocessing with Python, NumPy and Pandas

Internship task 1 - clean and preprocess a sample dataset and document every step.

## What's in this repo

```
make_dataset.py              builds the messy sample dataset (seeded, so repeatable)
preprocessing.py             the full cleaning pipeline
preprocessing_output.txt     console output from the last run
data/raw_customers.csv       dataset before cleaning (515 rows, 12 columns)
data/clean_customers.csv     after cleaning + encoding
data/clean_customers_scaled.csv   same, with numeric columns standardised
plots/                       charts from the EDA step
```

To run it yourself:

```
pip install -r requirements.txt
python make_dataset.py
python preprocessing.py
```

## About the dataset

I didn't want to just download Titanic like everyone else, so I generated a small customer
dataset and then deliberately broke it the way real data usually is broken: missing values,
duplicate rows, inconsistent spellings, impossible numbers and mixed date formats.

Columns: `customer_id, age, gender, city, education, annual_income, spending_score,
tenure_years, signup_date, membership, country, churned`.
The target I had in mind is `churned` (Yes/No).

## Steps

### 1. Loading and first look
Used `pd.read_csv`, then `.shape`, `.dtypes`, `.isnull().sum()` and `.describe()`.
Raw data is 515 rows x 12 columns. Missing values: age (31), city (15), education (26),
annual_income (41), spending_score (20).
`.describe()` already showed problems: age ranging from -5 to 250 and income up to 3,100,000
while the median is around 59,700.

### 2. Duplicate rows
`df.duplicated().sum()` gave 9 exact duplicates, removed with `drop_duplicates()`. 515 -> 506 rows.
This is done twice, see step 4 - 6 more duplicates were hiding because their dates were written
in a different format, so they only became exact copies after the dates were fixed.

### 3. Inconsistent text
- `gender` had 8 spellings (`Male, male, M, MALE `, `Female, female, F, ' Female'`).
  Stripped whitespace, lower-cased, took the first letter and mapped to Male/Female.
- `city` had both `Chennai` and `chennai` type duplicates. Fixed with `.str.strip().str.title()`,
  7 clean cities left.

### 4. Dates
`signup_date` was a string and had two formats (`2019-07-19` and `19/07/2019`).
Parsed each format separately with `errors="coerce"` and combined them. 0 dates failed.
A model can't use a raw date, so I pulled out `signup_year`.
After the dates matched I ran the duplicate check again and found 6 more copies (the rows were
identical except for the date format). Removed them: 506 -> 500 rows. Lesson: standardise the
formats first, then look for duplicates.

### 5. Impossible values
Ages of 140, -5 and 250 and one negative tenure were not real data. I set them to NaN so they
get handled together with the other missing values in the next step.

### 6. Missing values
| column | method | value |
|---|---|---|
| age | median | 38 |
| annual_income | median | 59,700 |
| spending_score | median | 48 |
| tenure_years | median | 4.9 |
| education | mode | Bachelors |
| city | mode | Bangalore |

Median instead of mean for numbers because the income column has huge outliers that would drag
the mean upward. Missing values after this step: 0.

### 7. Outliers
Used the IQR rule on `annual_income`: Q1 = 49,000, Q3 = 68,500, IQR = 19,500, so the allowed
range is 19,750 to 97,750. 5 values were outside it. I capped them to the limits instead of
deleting the rows, because the dataset is already small.

### 8. Feature selection
Dropped three columns:
- `country` - same value ("India") in every row, no information
- `customer_id` - just an identifier
- `signup_date` - replaced by `signup_year`

### 9. Encoding categorical variables
- `gender`, `churned` -> 0/1
- `education` (High School < Bachelors < Masters < PhD) and `membership` (Basic < Silver < Gold)
  -> ordinal numbers, since the order means something
- `city` -> one-hot with `pd.get_dummies` since cities have no order

Result: 500 rows x 16 columns, saved as `data/clean_customers.csv`.

### 10. Scaling
Standardised the numeric columns (age, income, spending_score, tenure, signup_year) with
z-score: `(x - mean) / std`, written with plain Pandas/NumPy so I could see the maths.
Mean is 0 and std is 1 for each afterwards. I also tried min-max on income as a comparison
(range 0 to 1). Saved as `data/clean_customers_scaled.csv`.
Scaling matters for distance-based models (KNN, SVM) and gradient descent, less so for trees.

### 11. Exploratory data analysis
- Overall churn rate: 19.2%
- Churn by membership: Basic 17.2%, Silver 21.1%, Gold 19.1% - not much difference
- Correlation with churn: `spending_score` (-0.22) and `tenure_years` (-0.15) are the strongest.
  Everything else is close to zero.

Plots are in `plots/`: income before vs after, age histogram, correlation matrix,
churn by membership.

## What I learned
- Always look at `.describe()` first, it caught the age of 250 immediately.
- The order of steps matters: fix invalid values before filling missing ones, clean
  text before encoding (otherwise `male` and `Male` become two categories), and fix
  formats before checking duplicates.
- Cap or fill instead of dropping when the dataset is small.
- Since this data is synthetic the weak correlations are expected; the point of the task
  was the process, not the model.

## Tools
Python 3, Pandas, NumPy, Matplotlib
