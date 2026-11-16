-- =============================================================================
-- sql/analysis.sql
-- Two-Wheeler Loan Risk Analytics - SQL Analysis
-- Run from Python via src/run_sql.py
-- =============================================================================

-- ─── Query 1: Overall default rate ───────────────────────────────────────────
-- Gives the headline KPI: what fraction of customers defaulted?
SELECT
    COUNT(*)                        AS total_customers,
    SUM(defaulted)                  AS total_defaults,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct
FROM loans;


-- ─── Query 2: Default rate by city tier ──────────────────────────────────────
-- Which city tier has the highest credit risk?
-- HAVING filters to tiers with more than 100 customers (removes noise).
SELECT
    city_tier,
    COUNT(*)                        AS total_customers,
    SUM(defaulted)                  AS defaults,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct,
    ROUND(AVG(monthly_income), 0)   AS avg_monthly_income
FROM loans
GROUP BY city_tier
HAVING COUNT(*) > 100
ORDER BY default_rate_pct DESC;


-- ─── Query 3: Default rate by employment type ─────────────────────────────────
-- Self-employed and daily-wage workers tend to have more volatile income.
SELECT
    employment_type,
    COUNT(*)                        AS total_customers,
    SUM(defaulted)                  AS defaults,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct,
    ROUND(AVG(emi), 0)              AS avg_emi,
    ROUND(AVG(credit_score), 0)     AS avg_credit_score
FROM loans
GROUP BY employment_type
HAVING COUNT(*) > 50
ORDER BY default_rate_pct DESC;


-- ─── Query 4: CASE WHEN — EMI-to-income risk buckets ─────────────────────────
-- Classify customers by how much of their income goes toward EMI.
-- EMI burden > 40% is considered "High Risk" in NBFC underwriting.
SELECT
    CASE
        WHEN emi * 1.0 / COALESCE(monthly_income, 1) < 0.20 THEN 'Low   (<20%)'
        WHEN emi * 1.0 / COALESCE(monthly_income, 1) < 0.40 THEN 'Medium (20-40%)'
        WHEN emi * 1.0 / COALESCE(monthly_income, 1) < 0.60 THEN 'High  (40-60%)'
        ELSE                                                       'Very High (>60%)'
    END AS emi_income_bucket,
    COUNT(*)                        AS customers,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct,
    ROUND(AVG(loan_amount), 0)      AS avg_loan_amount
FROM loans
WHERE monthly_income IS NOT NULL
GROUP BY emi_income_bucket
ORDER BY default_rate_pct DESC;


-- ─── Query 5: CTE — Top 10% riskiest customers (window function: NTILE) ───────
-- Uses a CTE + NTILE to rank all customers into 10 equal buckets by
-- a composite risk score. Bucket 10 = highest risk (top 10%).
WITH risk_scored AS (
    SELECT
        customer_id,
        employment_type,
        city_tier,
        COALESCE(monthly_income, 0)         AS monthly_income,
        credit_score,
        late_payments_last_12m,
        emi * 1.0 / COALESCE(monthly_income, 1) AS emi_to_income,
        defaulted,
        -- Composite risk score (higher = riskier)
        (  (1.0 - COALESCE(credit_score, 600) / 900.0) * 40
         + late_payments_last_12m * 5
         + CASE WHEN emi * 1.0 / COALESCE(monthly_income, 1) > 0.4 THEN 20 ELSE 0 END
         + CASE WHEN employment_type = 'Daily-Wage' THEN 10 ELSE 0 END
         + CASE WHEN employment_type = 'Student'    THEN 15 ELSE 0 END
        ) AS composite_risk_score
    FROM loans
),
bucketed AS (
    SELECT *,
        NTILE(10) OVER (ORDER BY composite_risk_score DESC) AS risk_decile
    FROM risk_scored
)
SELECT
    risk_decile,
    COUNT(*)                        AS customers,
    ROUND(AVG(defaulted) * 100, 2)  AS actual_default_rate_pct,
    ROUND(AVG(composite_risk_score), 1) AS avg_risk_score,
    ROUND(AVG(monthly_income), 0)   AS avg_income,
    ROUND(AVG(credit_score), 0)     AS avg_credit_score
FROM bucketed
GROUP BY risk_decile
ORDER BY risk_decile;


-- ─── Query 6: Subquery — Customers above-average EMI burden who defaulted ─────
-- Finds customers whose EMI is higher than the portfolio average,
-- and checks how many of them defaulted.
SELECT
    employment_type,
    COUNT(*)                        AS high_emi_customers,
    SUM(defaulted)                  AS defaults,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct
FROM loans
WHERE emi > (SELECT AVG(emi) FROM loans)
GROUP BY employment_type
ORDER BY default_rate_pct DESC;


-- ─── Query 7: Monthly default trend ──────────────────────────────────────────
-- Computes month-wise default rate to spot seasonal patterns or
-- portfolio quality trends over time.
SELECT
    SUBSTR(loan_start_date, 1, 7)   AS month,       -- 'YYYY-MM'
    COUNT(*)                        AS new_loans,
    SUM(defaulted)                  AS defaults,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct
FROM loans
GROUP BY month
ORDER BY month;


-- ─── Query 8: RANK — City-tier × employment default leaderboard ───────────────
-- Uses RANK() window function to rank each city-tier/employment combination
-- by default rate, so the NBFC can target high-risk segments first.
SELECT
    city_tier,
    employment_type,
    COUNT(*)                        AS customers,
    ROUND(AVG(defaulted) * 100, 2)  AS default_rate_pct,
    RANK() OVER (
        ORDER BY AVG(defaulted) DESC
    )                               AS risk_rank
FROM loans
GROUP BY city_tier, employment_type
HAVING COUNT(*) > 30
ORDER BY risk_rank;
