"""
predict.py
----------
Standalone CLI / module inference utility for Two-Wheeler Loan Default Prediction.
Loads models/best_model.joblib and evaluates any applicant in real time.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join("models", "best_model.joblib")

def load_inference_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 'python src/train_models.py' first.")
    return joblib.load(MODEL_PATH)

def predict_applicant(applicant_dict):
    """
    Evaluates applicant risk and returns probability, risk tier, and action.
    """
    artifacts = load_inference_artifacts()
    model = artifacts["model"]
    threshold = artifacts.get("threshold", 0.30)
    
    # Calculate derived features if not present
    age = applicant_dict.get("age", 30)
    monthly_income = applicant_dict.get("monthly_income", 35000)
    credit_score = applicant_dict.get("credit_score", 650)
    existing_loans = applicant_dict.get("existing_loans", 1)
    loan_amount = applicant_dict.get("loan_amount", 90000)
    tenure_months = applicant_dict.get("tenure_months", 36)
    interest_rate = applicant_dict.get("interest_rate", 14.5)
    vehicle_price = applicant_dict.get("vehicle_price", 110000)
    down_payment_pct = applicant_dict.get("down_payment_pct", 18.0)
    late_payments_last_12m = applicant_dict.get("late_payments_last_12m", 0)
    employment_type = applicant_dict.get("employment_type", "Salaried")
    city_tier = applicant_dict.get("city_tier", "Tier-2")
    loan_year = applicant_dict.get("loan_year", 2023)
    loan_month = applicant_dict.get("loan_month", 6)

    # Calculate monthly EMI using reducing balance
    monthly_rate = (interest_rate / 100) / 12
    if monthly_rate > 0 and tenure_months > 0:
        emi = (loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1)
    else:
        emi = loan_amount / max(tenure_months, 1)

    emi_to_income = round(emi / max(monthly_income, 1), 4)
    ltv_ratio = round(loan_amount / max(vehicle_price, 1), 4)
    late_pmt_flag = 1 if late_payments_last_12m > 0 else 0
    high_emi_burden = 1 if emi_to_income > 0.40 else 0

    # Categorical bins
    if monthly_income <= 15000:
        income_band = "Very Low"
    elif monthly_income <= 25000:
        income_band = "Low"
    elif monthly_income <= 40000:
        income_band = "Medium"
    elif monthly_income <= 60000:
        income_band = "High"
    else:
        income_band = "Very High"

    if credit_score <= 580:
        credit_band = "Poor"
    elif credit_score <= 650:
        credit_band = "Fair"
    elif credit_score <= 720:
        credit_band = "Good"
    elif credit_score <= 790:
        credit_band = "Very Good"
    else:
        credit_band = "Excellent"

    input_df = pd.DataFrame([{
        "age": age,
        "monthly_income": monthly_income,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "interest_rate": interest_rate,
        "emi": emi,
        "vehicle_price": vehicle_price,
        "down_payment_pct": down_payment_pct,
        "late_payments_last_12m": late_payments_last_12m,
        "emi_to_income": emi_to_income,
        "ltv_ratio": ltv_ratio,
        "late_pmt_flag": late_pmt_flag,
        "high_emi_burden": high_emi_burden,
        "loan_year": loan_year,
        "loan_month": loan_month,
        "employment_type": employment_type,
        "city_tier": city_tier,
        "income_band": income_band,
        "credit_band": credit_band
    }])

    prob_default = float(model.predict_proba(input_df)[0, 1])

    if prob_default < 0.20:
        risk_tier = "Low"
        underwriting_decision = "APPROVE (Standard Terms)"
        recommended_action = "Fast-track approval. Eligible for best pricing & cross-sell offers."
    elif prob_default < 0.45:
        risk_tier = "Medium"
        underwriting_decision = "CONDITIONAL APPROVAL / REVIEW"
        recommended_action = "Mandate e-NACH auto-debit, verify salary bank statements, max 75% LTV."
    else:
        risk_tier = "High"
        underwriting_decision = "REJECT / ESCALATE FOR CO-BORROWER"
        recommended_action = "Require credit-worthy co-applicant or increase down payment to 35%+."

    return {
        "default_probability": round(prob_default, 4),
        "risk_tier": risk_tier,
        "decision": underwriting_decision,
        "recommendation": recommended_action,
        "calculated_emi": round(emi, 2),
        "emi_to_income_ratio": round(emi_to_income * 100, 2),
        "ltv_pct": round(ltv_ratio * 100, 2)
    }

if __name__ == "__main__":
    print("=" * 70)
    print("  Two-Wheeler Loan Credit Risk Scorer (CLI Evaluation)")
    print("=" * 70)

    # Test cases:
    cases = [
        {
            "name": "Applicant A - Prime Salaried",
            "data": {
                "age": 34, "monthly_income": 52000, "credit_score": 740, "employment_type": "Salaried",
                "city_tier": "Tier-1", "loan_amount": 95000, "tenure_months": 24, "interest_rate": 12.5,
                "vehicle_price": 130000, "down_payment_pct": 27.0, "late_payments_last_12m": 0, "existing_loans": 0
            }
        },
        {
            "name": "Applicant B - Stretched Daily-Wage",
            "data": {
                "age": 28, "monthly_income": 16000, "credit_score": 560, "employment_type": "Daily-Wage",
                "city_tier": "Tier-3", "loan_amount": 105000, "tenure_months": 36, "interest_rate": 18.0,
                "vehicle_price": 120000, "down_payment_pct": 12.5, "late_payments_last_12m": 3, "existing_loans": 2
            }
        }
    ]

    for c in cases:
        print(f"\n--- Testing: {c['name']} ---")
        res = predict_applicant(c["data"])
        for k, v in res.items():
            print(f"  {k:22}: {v}")
    print("\n" + "=" * 70)
