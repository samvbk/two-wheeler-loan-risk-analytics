"""
generate_data.py
----------------
Generates a realistic synthetic dataset of 20,000 two-wheeler loan customers
for the "Two-Wheeler Loan Risk Analytics" project.

Design decisions:
- Fixed random seed (42) for full reproducibility
- Default rate ~12-15% built from realistic business logic
- Deliberate missing values and outliers for meaningful data cleaning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 20_000   # number of customers

# ── Helper: random dates ───────────────────────────────────────────────────────
def random_dates(start="2020-01-01", end="2023-12-31", n=N):
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    delta    = (end_dt - start_dt).days
    return [start_dt + timedelta(days=random.randint(0, delta)) for _ in range(n)]

# ── 1. Customer demographics ───────────────────────────────────────────────────
customer_id      = [f"CUST{str(i).zfill(6)}" for i in range(1, N + 1)]
age              = np.random.normal(32, 8, N).clip(18, 65).astype(int)

employment_types = ["Salaried", "Self-Employed", "Daily-Wage", "Student"]
employment_probs = [0.52, 0.28, 0.15, 0.05]
employment_type  = np.random.choice(employment_types, N, p=employment_probs)

city_tiers       = ["Tier-1", "Tier-2", "Tier-3"]
city_probs       = [0.30, 0.45, 0.25]
city_tier        = np.random.choice(city_tiers, N, p=city_probs)

# Monthly income varies by employment type
income_map = {
    "Salaried":     lambda: np.random.normal(42_000, 15_000),
    "Self-Employed":lambda: np.random.normal(38_000, 18_000),
    "Daily-Wage":   lambda: np.random.normal(18_000,  5_000),
    "Student":      lambda: np.random.normal(10_000,  3_000),
}
monthly_income = np.array([
    max(5_000, income_map[e]())
    for e in employment_type
]).round(0)

# Credit score: correlated with income / employment
base_score = (monthly_income / 100_000) * 200 + np.random.normal(550, 80, N)
credit_score = base_score.clip(300, 900).round(0).astype(int)

existing_loans = np.random.choice([0, 1, 2, 3, 4], N, p=[0.40, 0.30, 0.18, 0.08, 0.04])

# ── 2. Loan characteristics ────────────────────────────────────────────────────
vehicle_price    = np.random.choice(
    [75_000, 90_000, 1_10_000, 1_30_000, 1_60_000, 2_00_000, 2_50_000],
    N, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03]
)
down_payment_pct = np.random.beta(2, 5, N) * 40 + 5    # 5% – 45%
down_payment_pct = down_payment_pct.round(1)

loan_amount = (vehicle_price * (1 - down_payment_pct / 100)).round(0)

tenure_choices  = [12, 18, 24, 36, 48, 60]
tenure_months   = np.random.choice(tenure_choices, N, p=[0.05, 0.10, 0.25, 0.30, 0.20, 0.10])

# Interest rate: higher for risky profiles
base_rate = 14.0
rate_adj  = (
    - (credit_score - 600) / 100 * 2          # better credit → lower rate
    + (existing_loans / 4) * 2                 # more loans → higher rate
    + np.where(employment_type == "Daily-Wage", 2.5, 0)
    + np.where(employment_type == "Student",    3.0, 0)
    + np.random.normal(0, 1.5, N)
)
interest_rate = (base_rate + rate_adj).clip(10, 28).round(2)

# EMI calculation: standard reducing balance formula
monthly_rate = interest_rate / 100 / 12
emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months /
       ((1 + monthly_rate) ** tenure_months - 1)).round(0)

# Past late payment behaviour (12-month lookback)
late_base = (
    np.where(credit_score < 600, 3, 0)
    + np.where(employment_type == "Daily-Wage", 2, 0)
    + np.where(existing_loans >= 3, 1.5, 0)
    + np.random.poisson(0.3, N)
).clip(0, 12)
late_payments_last_12m = late_base.astype(int)

loan_start_date = random_dates()

# ── 3. Build target: defaulted ─────────────────────────────────────────────────
# Log-odds from business-meaningful factors
emi_to_income = emi / monthly_income.clip(1)

log_odds = (
    -4.5                                                      # intercept (sets ~12% base rate)
    + 3.5  * (emi_to_income > 0.4).astype(float)             # EMI burden > 40%
    + 1.8  * (emi_to_income > 0.6).astype(float)             # severe burden
    - 0.015 * (credit_score - 600)                            # credit score (continuous)
    + 0.35 * late_payments_last_12m                           # late payments
    + 0.6  * (down_payment_pct < 10).astype(float)            # very low down payment
    - 0.4  * (down_payment_pct > 25).astype(float)            # healthy down payment
    + 0.5  * (employment_type == "Daily-Wage").astype(float)
    + 0.7  * (employment_type == "Student").astype(float)
    - 0.3  * (city_tier == "Tier-1").astype(float)
    + 0.2  * (existing_loans >= 3).astype(float)
    + np.random.normal(0, 0.5, N)                             # noise
)

prob_default = 1 / (1 + np.exp(-log_odds))
defaulted    = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"Default rate: {defaulted.mean():.2%}  (target: 12-15%)")

# ── 4. Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "customer_id"          : customer_id,
    "age"                  : age,
    "monthly_income"       : monthly_income,
    "employment_type"      : employment_type,
    "city_tier"            : city_tier,
    "credit_score"         : credit_score,
    "existing_loans"       : existing_loans,
    "loan_amount"          : loan_amount,
    "tenure_months"        : tenure_months,
    "interest_rate"        : interest_rate,
    "emi"                  : emi,
    "vehicle_price"        : vehicle_price,
    "down_payment_pct"     : down_payment_pct,
    "late_payments_last_12m": late_payments_last_12m,
    "loan_start_date"      : [d.strftime("%Y-%m-%d") for d in loan_start_date],
    "defaulted"            : defaulted,
})

# ── 5. Inject deliberate missing values (~3-5%) ───────────────────────────────
miss_cols = ["monthly_income", "credit_score", "down_payment_pct", "interest_rate"]
for col in miss_cols:
    mask = np.random.choice([True, False], N, p=[0.04, 0.96])
    df.loc[mask, col] = np.nan

print(f"Missing values injected:\n{df[miss_cols].isnull().sum()}")

# ── 6. Inject deliberate outliers (~0.5%) ─────────────────────────────────────
outlier_idx = np.random.choice(N, size=int(N * 0.005), replace=False)
df.loc[outlier_idx[:30], "monthly_income"]  = np.random.uniform(500_000, 1_000_000, 30)
df.loc[outlier_idx[30:60], "loan_amount"]   = np.random.uniform(800_000, 1_200_000, 30)
df.loc[outlier_idx[60:80], "credit_score"]  = np.random.uniform(900, 999, 20)   # impossible scores
df.loc[outlier_idx[80:],   "emi"]           = np.random.uniform(50_000, 100_000, len(outlier_idx[80:]))

print("Outliers injected.")

# ── 7. Save ───────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "two_wheeler_loans.csv")
df.to_csv(out_path, index=False)
print(f"\nDataset saved -> {out_path}")
print(f"Shape: {df.shape}")
print(df.head(3).to_string())
