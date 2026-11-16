"""
app.py
------
Streamlit Web Application: Two-Wheeler Loan Risk Analytics
Features 3 tabs:
1. Loan Default Predictor (Live Underwriting Simulator)
2. Borrower Segmentation & Customer Insights
3. Delinquency Watchlist & Operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Two-Wheeler Loan Risk Analytics | NBFC Underwriting & Segmentation",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #000000;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .high-risk {
        background-color: #000000;
        border-left: 5px solid #d32f2f;
        padding: 12px;
        border-radius: 6px;
    }
    .med-risk {
        background-color: #000000;
        border-left: 5px solid #f57c00;
        padding: 12px;
        border-radius: 6px;
    }
    .low-risk {
        background-color: #000000;
        border-left: 5px solid #388e3c;
        padding: 12px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "best_model.joblib")
    seg_path = os.path.join("models", "segmentation.joblib")
    
    model_bundle = joblib.load(model_path) if os.path.exists(model_path) else None
    seg_bundle = joblib.load(seg_path) if os.path.exists(seg_path) else None
    return model_bundle, seg_bundle

@st.cache_data
def load_data():
    wl_path = os.path.join("reports", "watchlist_top500.csv")
    prof_path = os.path.join("reports", "cluster_profiles.csv")
    full_clust_path = os.path.join("data", "loans_with_clusters.csv")
    
    wl_df = pd.read_csv(wl_path) if os.path.exists(wl_path) else pd.DataFrame()
    prof_df = pd.read_csv(prof_path) if os.path.exists(prof_path) else pd.DataFrame()
    clust_df = pd.read_csv(full_clust_path) if os.path.exists(full_clust_path) else pd.DataFrame()
    return wl_df, prof_df, clust_df

model_bundle, seg_bundle = load_models()
watchlist_df, profile_df, clust_df = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/motorcycle.png", width=64)
st.sidebar.title("NBFC Risk Portal")
st.sidebar.markdown("**Portfolio Risk & Borrower Intelligence**")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status:**
- Model: `Logistic Regression (Balanced)`
- Validation ROC-AUC: **0.9195**
- Optimal Threshold: **0.30**
- Portfolio Size: **20,000 Loans**
- Baseline Default: **11.36%**
""")
st.sidebar.markdown("---")
st.sidebar.caption("Two-Wheeler Loan Risk Analytics v1.0.0")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏍️ Two-Wheeler Loan Risk Analytics Platform")
st.markdown("Automated Credit Risk Assessment, Customer Segmentation, and Early Delinquency Surveillance")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Underwriting Simulator (Default Predictor)",
    "👥 Borrower Segmentation & Strategy",
    "🚨 Delinquency Watchlist & Operations"
])

# ==============================================================================
# TAB 1: DEFAULT PREDICTION
# ==============================================================================
with tab1:
    st.subheader("Real-Time Loan Applicant Risk Scoring")
    st.markdown("Enter applicant financial and loan parameters to assess default risk and generate underwriting recommendations.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Applicant Demographics")
        age = st.slider("Applicant Age", min_value=18, max_value=65, value=32)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Daily-Wage", "Student"])
        city_tier = st.selectbox("City Tier", ["Tier-1", "Tier-2", "Tier-3"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=500000, value=35000, step=2500)
        
    with col2:
        st.markdown("##### 💳 Credit & Bureau History")
        credit_score = st.slider("Credit Bureau Score (CIBIL)", min_value=300, max_value=900, value=650)
        existing_loans = st.number_input("Existing Active Loans", min_value=0, max_value=10, value=1)
        late_payments = st.number_input("Late Payments in Last 12M", min_value=0, max_value=12, value=0)
        
    with col3:
        st.markdown("##### 🏍️ Two-Wheeler Loan Terms")
        vehicle_price = st.selectbox("Vehicle On-Road Price (₹)", [75000, 90000, 110000, 130000, 160000, 200000, 250000], index=2)
        down_payment_pct = st.slider("Down Payment (%)", min_value=5.0, max_value=50.0, value=20.0, step=0.5)
        tenure_months = st.selectbox("Loan Tenure (Months)", [12, 18, 24, 36, 48, 60], index=2)
        interest_rate = st.slider("Interest Rate (p.a. %)", min_value=10.0, max_value=28.0, value=14.5, step=0.25)

    # Derived calculations
    loan_amount = round(vehicle_price * (1 - (down_payment_pct / 100.0)), 2)
    monthly_rate = (interest_rate / 100.0) / 12.0
    if monthly_rate > 0 and tenure_months > 0:
        emi = round((loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1), 2)
    else:
        emi = round(loan_amount / max(tenure_months, 1), 2)
    
    emi_to_income = emi / max(monthly_income, 1)
    ltv_ratio = loan_amount / max(vehicle_price, 1)

    st.markdown("---")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Calculated Loan Amount", f"₹{loan_amount:,.0f}")
    mcol2.metric("Monthly EMI", f"₹{emi:,.0f}")
    mcol3.metric("EMI-to-Income (FOIR)", f"{emi_to_income * 100:.1f}%", 
                 delta="High Risk" if emi_to_income > 0.40 else "Healthy", delta_color="inverse")
    mcol4.metric("Loan-to-Value (LTV)", f"{ltv_ratio * 100:.1f}%")

    if st.button("Evaluate Applicant Risk", type="primary", use_container_width=True):
        if model_bundle is not None:
            # Prepare feature frame matching model training
            late_pmt_flag = 1 if late_payments > 0 else 0
            high_emi_burden = 1 if emi_to_income > 0.40 else 0
            
            # Categorical bands
            if monthly_income <= 15000: inc_band = "Very Low"
            elif monthly_income <= 25000: inc_band = "Low"
            elif monthly_income <= 40000: inc_band = "Medium"
            elif monthly_income <= 60000: inc_band = "High"
            else: inc_band = "Very High"

            if credit_score <= 580: cred_band = "Poor"
            elif credit_score <= 650: cred_band = "Fair"
            elif credit_score <= 720: cred_band = "Good"
            elif credit_score <= 790: cred_band = "Very Good"
            else: cred_band = "Excellent"

            applicant_row = pd.DataFrame([{
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
                "late_payments_last_12m": late_payments,
                "emi_to_income": round(emi_to_income, 4),
                "ltv_ratio": round(ltv_ratio, 4),
                "late_pmt_flag": late_pmt_flag,
                "high_emi_burden": high_emi_burden,
                "loan_year": 2023,
                "loan_month": 6,
                "employment_type": employment_type,
                "city_tier": city_tier,
                "income_band": inc_band,
                "credit_band": cred_band
            }])

            model = model_bundle["model"]
            prob = float(model.predict_proba(applicant_row)[0, 1])

            st.markdown("### Risk Evaluation Results")
            rcol1, rcol2 = st.columns([1, 2])

            with rcol1:
                st.metric("Predicted Default Probability", f"{prob * 100:.2f}%")
                
                if prob < 0.20:
                    tier_class = "low-risk"
                    tier_name = "LOW RISK"
                    badge = "🟢"
                elif prob < 0.45:
                    tier_class = "med-risk"
                    tier_name = "MEDIUM RISK"
                    badge = "🟡"
                else:
                    tier_class = "high-risk"
                    tier_name = "HIGH RISK"
                    badge = "🔴"

                st.markdown(f"""
                <div class="{tier_class}">
                    <h4>{badge} {tier_name}</h4>
                    Decision Threshold: <b>0.30</b>
                </div>
                """, unsafe_allow_html=True)

            with rcol2:
                st.markdown("##### Underwriting Policy Recommendations")
                if prob < 0.20:
                    st.success("""
                    - **Recommendation:** APPROVE (Fast-track)
                    - **Pricing:** Offer preferential interest rate discount (-0.50%).
                    - **Next Best Action:** Cross-sell two-wheeler extended warranty and comprehensive insurance.
                    """)
                elif prob < 0.45:
                    st.warning("""
                    - **Recommendation:** CONDITIONAL APPROVAL / MANUAL REVIEW
                    - **Mitigation Terms:** Mandate auto-debit (e-NACH) with proof of salary account.
                    - **Policy Adjustments:** Cap LTV at 75% or shorten tenure to 24 months to lower overall risk exposure.
                    """)
                else:
                    st.error("""
                    - **Recommendation:** REJECT OR REQUIRE CO-APPLICANT
                    - **Risk Drivers:** Stretched EMI-to-income ratio, low credit score, or previous late payment history.
                    - **Restructuring Option:** Request 35%+ down payment or a creditworthy salaried family member as co-borrower.
                    """)
        else:
            st.error("Model artifacts not found. Please train models using `python src/train_models.py`.")


# ==============================================================================
# TAB 2: BORROWER SEGMENTATION
# ==============================================================================
with tab2:
    st.subheader("👥 Data-Driven Customer Segmentation (K-Means Clustering)")
    st.markdown("Borrowers are grouped into 4 distinct segments based on financial profile, leverage, and past credit discipline.")

    if not profile_df.empty:
        # Display cluster cards
        card_cols = st.columns(4)
        cluster_names = [
            ("Conservative Low-LTV", "2.0% Default", "Cluster 0", "#2ecc71"),
            ("Prime Salaried Growth", "1.0% Default", "Cluster 1", "#27ae60"),
            ("High-Risk Over-Leveraged", "61.0% Default", "Cluster 2", "#e74c3c"),
            ("Vulnerable Moderate-Risk", "17.0% Default", "Cluster 3", "#f39c12")
        ]

        for i, (name, def_rate, c_id, color) in enumerate(cluster_names):
            with card_cols[i]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-top: 4px solid {color}; padding: 10px; border-radius: 6px; text-align: center;">
                    <span style="font-size: 12px; color: gray;">{c_id}</span>
                    <h5 style="margin: 5px 0;">{name}</h5>
                    <strong style="color: {color};">{def_rate}</strong>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Detailed Segment Profiles & Underwriting Strategies")
        
        display_prof = profile_df.copy()
        if "cluster" in display_prof.columns:
            display_prof = display_prof.set_index("cluster")
        
        cols_to_show = [c for c in [
            "cluster_name", "n_customers", "default_rate_pct", "monthly_income",
            "credit_score", "loan_amount", "emi_to_income", "down_payment_pct",
            "late_payments_last_12m", "recommended_action"
        ] if c in display_prof.columns]
        
        st.dataframe(display_prof[cols_to_show].style.format({
            "monthly_income": "₹{:,.0f}",
            "loan_amount": "₹{:,.0f}",
            "credit_score": "{:.0f}",
            "default_rate_pct": "{:.1f}%",
            "emi_to_income": "{:.2f}",
            "down_payment_pct": "{:.1f}%",
            "late_payments_last_12m": "{:.1f}"
        }), use_container_width=True)

        # Visualizations
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("##### PCA Projection of Segments")
            if os.path.exists("reports/11_pca_clusters.png"):
                st.image("reports/11_pca_clusters.png", use_container_width=True)
            else:
                st.info("PCA chart available in reports/11_pca_clusters.png")

        with scol2:
            st.markdown("##### Relative Feature Heatmap Across Clusters")
            if os.path.exists("reports/12_cluster_heatmap.png"):
                st.image("reports/12_cluster_heatmap.png", use_container_width=True)
            else:
                st.info("Cluster heatmap available in reports/12_cluster_heatmap.png")

    else:
        st.warning("Segmentation data not found. Please run `python src/segmentation.py`.")


# ==============================================================================
# TAB 3: DELINQUENCY WATCHLIST
# ==============================================================================
with tab3:
    st.subheader("🚨 Early Delinquency Watchlist & Collections Operations")
    st.markdown("Proactive surveillance list of highest-risk borrowers to trigger preventative collections outreach before loans become non-performing (NPA).")

    if not watchlist_df.empty:
        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            sel_tier = st.multiselect("Risk Tier", options=watchlist_df["risk_label"].dropna().unique(), default=watchlist_df["risk_label"].dropna().unique())
        with fcol2:
            sel_emp = st.multiselect("Employment Type", options=watchlist_df["employment_type"].dropna().unique(), default=watchlist_df["employment_type"].dropna().unique())
        with fcol3:
            min_prob = st.slider("Minimum Default Probability", 0.0, 1.0, 0.40, 0.05)

        filtered_wl = watchlist_df[
            (watchlist_df["risk_label"].isin(sel_tier)) &
            (watchlist_df["employment_type"].isin(sel_emp)) &
            (watchlist_df["default_probability"] >= min_prob)
        ].copy()

        # Summary KPIs
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        kcol1.metric("Flagged Accounts", f"{len(filtered_wl):,}")
        kcol2.metric("Total Exposure at Risk", f"₹{filtered_wl['loan_amount'].sum():,.0f}")
        kcol3.metric("Avg Default Probability", f"{filtered_wl['default_probability'].mean() * 100:.1f}%")
        kcol4.metric("Actual Default Rate in Group", f"{filtered_wl['defaulted'].mean() * 100:.1f}%")

        st.markdown("---")
        st.markdown("##### Priority Action Queue")
        
        # Color coding helper
        def highlight_risk(val):
            if val == "High":
                return "background-color: #ffcdd2"
            elif val == "Medium":
                return "background-color: #fff9c4"
            return "background-color: #c8e6c9"

        st.dataframe(
            filtered_wl[[
                "customer_id", "employment_type", "city_tier", "monthly_income",
                "loan_amount", "credit_score", "late_payments_last_12m",
                "emi_to_income", "default_probability", "risk_label"
            ]].style.format({
                "monthly_income": "₹{:,.0f}",
                "loan_amount": "₹{:,.0f}",
                "credit_score": "{:.0f}",
                "emi_to_income": "{:.2f}",
                "default_probability": "{:.2%}"
            }).map(highlight_risk, subset=["risk_label"]),
            use_container_width=True,
            height=400
        )

        csv = filtered_wl.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Watchlist (CSV) for Collections Team",
            data=csv,
            file_name="collections_priority_watchlist.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Watchlist data not found. Please run `python src/train_models.py`.")
