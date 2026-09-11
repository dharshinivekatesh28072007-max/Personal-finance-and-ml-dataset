"""Extended data-integrity and exploratory audit for the Personal Finance
ML dataset.
Matches the actual dataset schema:
user_id, age, gender, education_level, employment_status, job_title,
monthly_income_usd, monthly_expenses_usd, savings_usd, has_loan,
loan_type, loan_amount_usd, loan_term_months, monthly_emi_usd,
loan_interest_rate_pct, debt_to_income_ratio, credit_score,
savings_to_income_ratio, region, record_date
This is a larger, more thorough version of the original audit script.
Every quantitative claim comes from a function defined here so results
are reproducible and traceable back to a single line of code.
Sections covered:
    1.  Schema / shape checks
    2.  Uniqueness / duplicate checks
    3.  Missingness
    4.  Cardinality of categorical columns
    5.  Categorical distributions (region, employment, education, gender,
        job title, loan type)
    6.  Central tendency (mean/median/std) for numeric columns
    7.  Negative / out-of-range value checks
    8.  Cross-field financial sanity checks
    9.  Loan-specific consistency checks
    10. Outlier detection (IQR method)
    11. Savings-rate cross-validation
    12. Debt-to-income cross-validation against recomputed EMI/income
    13. Correlations between numeric variables
    14. Value range summary (min/max) for numeric columns
    15. Grouped averages (by region, employment status, education level)
    16. Age-band and credit-score-band breakdowns
    17. Date-field sanity checks
    18. Full report runner + pretty printer
"""
from __future__ import annotations
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
USER_ID = "user_id"
AGE = "age"
GENDER = "gender"
EDUCATION = "education_level"
EMPLOYMENT = "employment_status"
JOB_TITLE = "job_title"
INCOME = "monthly_income_usd"
EXPENSES = "monthly_expenses_usd"
SAVINGS = "savings_usd"
HAS_LOAN = "has_loan"
LOAN_TYPE = "loan_type"
LOAN_AMOUNT = "loan_amount_usd"
LOAN_TERM = "loan_term_months"
LOAN_EMI = "monthly_emi_usd"
LOAN_RATE = "loan_interest_rate_pct"
DTI = "debt_to_income_ratio"
CREDIT_SCORE = "credit_score"
SAVINGS_RATIO = "savings_to_income_ratio"
REGION = "region"
RECORD_DATE = "record_date"
NUMERIC_COLUMNS = [
    AGE,
    INCOME,
    EXPENSES,
    SAVINGS,
    LOAN_AMOUNT,
    LOAN_TERM,
    LOAN_EMI,
    LOAN_RATE,
    DTI,
    CREDIT_SCORE,
    SAVINGS_RATIO,
]
CATEGORICAL_COLUMNS = [
    GENDER,
    EDUCATION,
    EMPLOYMENT,
    JOB_TITLE,
    HAS_LOAN,
    LOAN_TYPE,
    REGION,
]
# ---------------------------------------------------------------------------
# 1. Schema / shape checks
# ---------------------------------------------------------------------------
def shape(df: pd.DataFrame) -> dict[str, int]:
    """Row and column counts."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }
def column_types(df: pd.DataFrame) -> dict[str, str]:
    """Pandas-inferred dtype for every column."""
    return {column: str(dtype) for column, dtype in df.dtypes.items()}
def schema_check(df: pd.DataFrame) -> dict[str, object]:
    """Confirm every expected column is present, and flag any extras."""
    expected = {
        USER_ID, AGE, GENDER, EDUCATION, EMPLOYMENT, JOB_TITLE,
        INCOME, EXPENSES, SAVINGS, HAS_LOAN, LOAN_TYPE, LOAN_AMOUNT,
        LOAN_TERM, LOAN_EMI, LOAN_RATE, DTI, CREDIT_SCORE,
        SAVINGS_RATIO, REGION, RECORD_DATE,
    }
    actual = set(df.columns)
    return {
        "missing_expected_columns": sorted(expected - actual),
        "unexpected_extra_columns": sorted(actual - expected),
    }
# ---------------------------------------------------------------------------
# 2. Uniqueness / duplicate checks
# ---------------------------------------------------------------------------
def uniqueness(df: pd.DataFrame) -> dict[str, int]:
    """Check duplicate rows and duplicate user IDs."""
    return {
        "total_rows": int(len(df)),
        "unique_rows": int(df.drop_duplicates().shape[0]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_user_ids": int(df[USER_ID].duplicated().sum()),
    }
def duplicate_user_id_list(df: pd.DataFrame) -> list[str]:
    """List the specific user_id values that appear more than once."""
    counts = df[USER_ID].value_counts()
    return sorted(counts[counts > 1].index.tolist())
# ---------------------------------------------------------------------------
# 3. Missingness
# ---------------------------------------------------------------------------
def missingness(df: pd.DataFrame) -> dict[str, int]:
    """Null count per column, restricted to columns that have any."""
    counts = df.isna().sum()
    return {
        column: int(count)
        for column, count in counts.items()
        if count > 0
    }
def missingness_pct(df: pd.DataFrame) -> dict[str, float]:
    """Percentage of rows missing each column, restricted to columns
    that have any nulls."""
    total = len(df)
    counts = df.isna().sum()
    return {
        column: round(float(count) / total * 100, 2)
        for column, count in counts.items()
        if count > 0
    }
# ---------------------------------------------------------------------------
# 4. Cardinality
# ---------------------------------------------------------------------------
def cardinalities(df: pd.DataFrame) -> dict[str, int]:
    """Distinct non-null values for each categorical column."""
    return {
        column: int(df[column].nunique(dropna=True))
        for column in CATEGORICAL_COLUMNS
    }
# ---------------------------------------------------------------------------
# 5. Categorical distributions
# ---------------------------------------------------------------------------
def category_distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Number of records per value of a categorical column."""
    counts = df[column].value_counts()
    return {str(value): int(count) for value, count in counts.items()}
def loan_type_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Number of records per loan type, restricted to loan holders."""
    counts = df.loc[df[HAS_LOAN] == "Yes", LOAN_TYPE].value_counts()
    return {str(value): int(count) for value, count in counts.items()}
def all_categorical_distributions(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Distributions for every categorical column in one call."""
    return {
        column: category_distribution(df, column)
        for column in CATEGORICAL_COLUMNS
    }
# ---------------------------------------------------------------------------
# 6. Central tendency
# ---------------------------------------------------------------------------
def central_tendency(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean, median and standard deviation for every numeric column."""
    result = {}
    for column in NUMERIC_COLUMNS:
        result[column] = {
            "mean": round(float(df[column].mean()), 2),
            "median": round(float(df[column].median()), 2),
            "std": round(float(df[column].std()), 2),
        }
    return result
def average_finance(df: pd.DataFrame) -> dict[str, float]:
    """Average income, expenses, savings, age and credit score
    (kept for backward compatibility with the original audit)."""
    return {
        "average_income": round(float(df[INCOME].mean()), 2),
        "average_expenses": round(float(df[EXPENSES].mean()), 2),
        "average_savings": round(float(df[SAVINGS].mean()), 2),
        "average_age": round(float(df[AGE].mean()), 2),
        "average_credit_score": round(float(df[CREDIT_SCORE].mean()), 2),
    }
# ---------------------------------------------------------------------------
# 7. Negative / out-of-range value checks
# ---------------------------------------------------------------------------
def negative_values(df: pd.DataFrame) -> dict[str, int]:
    """Count of negative values per numeric column (none should ever be
    negative in this dataset)."""
    return {
        column: int((df[column] < 0).sum())
        for column in NUMERIC_COLUMNS
    }
def out_of_range_values(df: pd.DataFrame) -> dict[str, int]:
    """Checks against known plausible ranges for specific fields."""
    return {
        "age_outside_18_100": int(((df[AGE] < 18) | (df[AGE] > 100)).sum()),
        "credit_score_outside_300_850": int(
            ((df[CREDIT_SCORE] < 300) | (df[CREDIT_SCORE] > 850)).sum()
        ),
        "loan_interest_rate_outside_0_40": int(
            ((df[LOAN_RATE] < 0) | (df[LOAN_RATE] > 40)).sum()
        ),
        "loan_term_outside_0_480_months": int(
            ((df[LOAN_TERM] < 0) | (df[LOAN_TERM] > 480)).sum()
        ),
    }
# ---------------------------------------------------------------------------
# 8. Cross-field financial sanity checks
# ---------------------------------------------------------------------------
def finance_sanity(df: pd.DataFrame) -> dict[str, int]:
    """Check basic financial-data consistency across fields."""
    return {
        "zero_or_negative_income": int((df[INCOME] <= 0).sum()),
        "negative_expenses": int((df[EXPENSES] < 0).sum()),
        "negative_savings": int((df[SAVINGS] < 0).sum()),
        "invalid_age": int(((df[AGE] < 0) | (df[AGE] > 120)).sum()),
        "invalid_credit_score": int(
            ((df[CREDIT_SCORE] < 300) | (df[CREDIT_SCORE] > 850)).sum()
        ),
        "expenses_greater_than_income": int((df[EXPENSES] > df[INCOME]).sum()),
        "expenses_equal_zero": int((df[EXPENSES] == 0).sum()),
        "income_below_500": int((df[INCOME] < 500).sum()),
    }
# ---------------------------------------------------------------------------
# 9. Loan-specific consistency checks
# ---------------------------------------------------------------------------
def loan_consistency(df: pd.DataFrame) -> dict[str, int]:
    """Cross-checks between the has_loan flag and the loan detail fields."""
    loan_rows = df[HAS_LOAN] == "Yes"
    no_loan_rows = df[HAS_LOAN] == "No"
    return {
        "has_loan_yes_but_zero_amount": int(
            (loan_rows & (df[LOAN_AMOUNT] <= 0)).sum()
        ),
        "has_loan_yes_but_zero_term": int(
            (loan_rows & (df[LOAN_TERM] <= 0)).sum()
        ),
        "has_loan_yes_but_zero_emi": int(
            (loan_rows & (df[LOAN_EMI] <= 0)).sum()
        ),
        "has_loan_yes_but_missing_loan_type": int(
            (loan_rows & df[LOAN_TYPE].isna()).sum()
        ),
        "has_loan_no_but_nonzero_amount": int(
            (no_loan_rows & (df[LOAN_AMOUNT] != 0)).sum()
        ),
        "has_loan_no_but_nonzero_emi": int(
            (no_loan_rows & (df[LOAN_EMI] != 0)).sum()
        ),
        "has_loan_no_but_loan_type_set": int(
            (no_loan_rows & df[LOAN_TYPE].notna() & (df[LOAN_TYPE] != "None")).sum()
        ),
        "loan_emi_times_term_exceeds_5x_amount": int(
            (
                loan_rows
                & ((df[LOAN_EMI] * df[LOAN_TERM]) > (df[LOAN_AMOUNT] * 5))
            ).sum()
        ),
    }
# ---------------------------------------------------------------------------
# 10. Outlier detection (IQR method)
# ---------------------------------------------------------------------------
def iqr_outliers(df: pd.DataFrame, column: str, k: float = 1.5) -> dict[str, object]:
    """Count and bound values outside [Q1 - k*IQR, Q3 + k*IQR] for one column."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outlier_mask = (df[column] < lower) | (df[column] > upper)
    return {
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "outlier_count": int(outlier_mask.sum()),
    }
def all_outliers(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """IQR-based outlier summary for the core financial columns."""
    columns = [INCOME, EXPENSES, SAVINGS, LOAN_AMOUNT, LOAN_EMI, CREDIT_SCORE]
    return {column: iqr_outliers(df, column) for column in columns}
# ---------------------------------------------------------------------------
# 11. Savings-rate cross-validation
# ---------------------------------------------------------------------------
def savings_rate(df: pd.DataFrame) -> dict[str, float]:
    """Compute savings / monthly income and compare against the
    dataset's own savings_to_income_ratio column."""
    computed_rate = np.where(
        df[INCOME] > 0,
        (df[SAVINGS] / df[INCOME]),
        np.nan,
    )
    return {
        "average_computed_savings_ratio": round(float(np.nanmean(computed_rate)), 2),
        "minimum_computed_savings_ratio": round(float(np.nanmin(computed_rate)), 2),
        "maximum_computed_savings_ratio": round(float(np.nanmax(computed_rate)), 2),
        "average_reported_savings_to_income_ratio": round(
            float(df[SAVINGS_RATIO].mean()), 2
        ),
        "max_abs_diff_computed_vs_reported": round(
            float(np.nanmax(np.abs(computed_rate - df[SAVINGS_RATIO]))), 2
        ),
    }
# ---------------------------------------------------------------------------
# 12. Debt-to-income cross-validation
# ---------------------------------------------------------------------------
def dti_cross_check(df: pd.DataFrame) -> dict[str, float]:
    """Recompute an approximate DTI (EMI / income) for loan holders and
    compare it against the dataset's reported debt_to_income_ratio."""
    loan_rows = df[df[HAS_LOAN] == "Yes"].copy()
    computed_dti = np.where(
        loan_rows[INCOME] > 0,
        (loan_rows[LOAN_EMI] / loan_rows[INCOME]) * 100,
        np.nan,
    )
    diff = np.abs(computed_dti - loan_rows[DTI])
    return {
        "loan_holder_count": int(len(loan_rows)),
        "average_computed_dti_pct": round(float(np.nanmean(computed_dti)), 2),
        "average_reported_dti_pct": round(float(loan_rows[DTI].mean()), 2),
        "average_abs_diff_pct": round(float(np.nanmean(diff)), 2),
        "max_abs_diff_pct": round(float(np.nanmax(diff)), 2),
    }
# ---------------------------------------------------------------------------
# 13. Correlations
# ---------------------------------------------------------------------------
def correlations(df: pd.DataFrame) -> dict[str, float]:
    """Pearson correlations between the core numeric finance variables."""
    columns = [INCOME, EXPENSES, SAVINGS, AGE, CREDIT_SCORE, DTI]
    matrix = df[columns].corr()
    result = {}
    for i, column1 in enumerate(columns):
        for column2 in columns[i + 1:]:
            result[f"{column1}~{column2}"] = round(
                float(matrix.loc[column1, column2]), 3
            )
    return result
# ---------------------------------------------------------------------------
# 14. Value range summary
# ---------------------------------------------------------------------------
def value_sanity(df: pd.DataFrame) -> dict[str, float | int]:
    """Min/max range checks on the core personal finance values."""
    return {
        "income_min": round(float(df[INCOME].min()), 2),
        "income_max": round(float(df[INCOME].max()), 2),
        "expenses_min": round(float(df[EXPENSES].min()), 2),
        "expenses_max": round(float(df[EXPENSES].max()), 2),
        "savings_min": round(float(df[SAVINGS].min()), 2),
        "savings_max": round(float(df[SAVINGS].max()), 2),
        "age_min": int(df[AGE].min()),
        "age_max": int(df[AGE].max()),
        "credit_score_min": int(df[CREDIT_SCORE].min()),
        "credit_score_max": int(df[CREDIT_SCORE].max()),
    }
# ---------------------------------------------------------------------------
# 15. Grouped averages
# ---------------------------------------------------------------------------
def grouped_averages(df: pd.DataFrame, group_column: str) -> dict[str, dict[str, float]]:
    """Average income, expenses, savings and credit score by a
    categorical grouping column."""
    grouped = df.groupby(group_column)[
        [INCOME, EXPENSES, SAVINGS, CREDIT_SCORE]
    ].mean().round(2)
    return {
        str(group): row.to_dict()
        for group, row in grouped.iterrows()
    }
def averages_by_region(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return grouped_averages(df, REGION)
def averages_by_employment(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return grouped_averages(df, EMPLOYMENT)
def averages_by_education(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return grouped_averages(df, EDUCATION)
# ---------------------------------------------------------------------------
# 16. Age-band and credit-score-band breakdowns
# ---------------------------------------------------------------------------
def age_band_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Bucket ages into 10-year bands and count records per band."""
    bins = [17, 25, 35, 45, 55, 65, 120]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    bands = pd.cut(df[AGE], bins=bins, labels=labels)
    counts = bands.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}
def credit_score_band_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Bucket credit scores into common risk bands."""
    bins = [299, 579, 669, 739, 799, 850]
    labels = ["Poor (300-579)", "Fair (580-669)", "Good (670-739)",
              "Very Good (740-799)", "Excellent (800-850)"]
    bands = pd.cut(df[CREDIT_SCORE], bins=bins, labels=labels)
    counts = bands.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}
# ---------------------------------------------------------------------------
# 17. Date-field sanity checks
# ---------------------------------------------------------------------------
def date_sanity(df: pd.DataFrame) -> dict[str, object]:
    """Check that record_date parses cleanly and falls in a sane range."""
    parsed = pd.to_datetime(df[RECORD_DATE], errors="coerce")
    return {
        "unparseable_dates": int(parsed.isna().sum()),
        "earliest_date": str(parsed.min().date()) if parsed.notna().any() else None,
        "latest_date": str(parsed.max().date()) if parsed.notna().any() else None,
        "records_before_2020": int((parsed < "2020-01-01").sum()),
        "records_after_today": int((parsed > pd.Timestamp.now()).sum()),
    }
# ---------------------------------------------------------------------------
# 18. Full report runner
# ---------------------------------------------------------------------------
def run_all(df: pd.DataFrame | None = None) -> dict[str, object]:
    """Run every audit check and return one combined results dictionary."""

    data = (
        df.copy()
        if df is not None
        else pd.read_csv("personal_finance.csv",encoding="utf-8-sig")
    )

    # Remove unwanted spaces from column names
    data.columns = data.columns.str.strip()

    return {
        "shape": shape(data),
        "column_types": column_types(data),
        "schema_check": schema_check(data),
        "uniqueness": uniqueness(data),
        "duplicate_user_ids": duplicate_user_id_list(data),
        "missingness": missingness(data),
        "missingness_pct": missingness_pct(data),
        "cardinalities": cardinalities(data),
        "categorical_distributions": all_categorical_distributions(data),
        "loan_type_distribution": loan_type_distribution(data),
        "central_tendency": central_tendency(data),
        "average_finance": average_finance(data),
        "negative_values": negative_values(data),
        "out_of_range_values": out_of_range_values(data),
        "finance_sanity": finance_sanity(data),
        "loan_consistency": loan_consistency(data),
        "outliers": all_outliers(data),
        "savings_rate": savings_rate(data),
        "dti_cross_check": dti_cross_check(data),
        "correlations": correlations(data),
        "value_sanity": value_sanity(data),
        "averages_by_region": averages_by_region(data),
        "averages_by_employment": averages_by_employment(data),
        "averages_by_education": averages_by_education(data),
        "age_band_distribution": age_band_distribution(data),
        "credit_score_band_distribution": credit_score_band_distribution(data),
        "date_sanity": date_sanity(data),
    }


def print_report(results: dict[str, object]) -> None:
    """Pretty-print the complete audit report."""

    print("\n" + "=" * 80)
    print("        PERSONAL FINANCE ML DATASET - EXTENDED AUDIT")
    print("=" * 80)

    for section, values in results.items():

        print("\n" + "-" * 80)
        print(f"SECTION: {section.upper()}")
        print("-" * 80)

        if isinstance(values, dict):

            for key, value in values.items():

                if isinstance(value, dict):

                    print(f"\n{key}:")

                    for sub_key, sub_value in value.items():

                        if isinstance(sub_value, dict):

                            print(f"  {sub_key}:")

                            for inner_key, inner_value in sub_value.items():
                                print(
                                    f"      {inner_key}: {inner_value}"
                                )

                        else:
                            print(
                                f"  {sub_key}: {sub_value}"
                            )

                elif isinstance(value, list):

                    print(f"{key}:")

                    if len(value) == 0:
                        print("  None")

                    else:
                        for item in value:
                            print(f"  - {item}")

                else:
                    print(f"{key}: {value}")

        elif isinstance(values, list):

            if len(values) == 0:
                print("None")

            else:
                for item in values:
                    print(f"- {item}")

        else:
            print(values)

    print("\n" + "=" * 80)
    print ("AUDIT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    results = run_all()
    print_report(results)