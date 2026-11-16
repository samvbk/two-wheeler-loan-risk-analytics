"""
segmentation.py
---------------
Part 5: Borrower Segmentation
- K-Means clustering with elbow + silhouette selection
- PCA visualisation
- Cluster profiling and business naming
- Tests whether cluster label improves default prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)

from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score, roc_auc_score
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OrdinalEncoder
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

SEED = 42
np.random.seed(SEED)

# ─── 1. Load Features ─────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join("data", "loans_features.csv"))
print(f"Loaded: {df.shape}")

# Features used for clustering (financial profile, not identity columns)
CLUSTER_FEATURES = [
    "age", "monthly_income", "credit_score", "existing_loans",
    "loan_amount", "interest_rate", "emi_to_income", "ltv_ratio",
    "down_payment_pct", "late_payments_last_12m"
]
# Drop rows with missing cluster features
df_clust = df[CLUSTER_FEATURES + ["defaulted", "customer_id"]].dropna().copy()
print(f"Rows for clustering: {len(df_clust)}")

X_clust = df_clust[CLUSTER_FEATURES]

# ─── 2. Scale Features ────────────────────────────────────────────────────────
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_clust)

# ─── 3. Elbow + Silhouette to Choose K ───────────────────────────────────────
print("\n--- Elbow + Silhouette Analysis ---")
K_RANGE  = range(2, 10)
inertias = []
sil_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000,
                                        random_state=SEED))
    print(f"  k={k}: inertia={km.inertia_:.0f}  silhouette={sil_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inertias, "bo-", lw=2, markersize=7)
axes[0].set_title("Elbow Method — Inertia vs k", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
for k, v in zip(K_RANGE, inertias):
    axes[0].annotate(f"{v:.0f}", (k, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7)

axes[1].plot(list(K_RANGE), sil_scores, "rs-", lw=2, markersize=7)
axes[1].set_title("Silhouette Score vs k", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score (higher = better)")
best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
axes[1].axvline(best_k, color="gray", linestyle="--",
                label=f"Best k={best_k} (sil={max(sil_scores):.3f})")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/10_elbow_silhouette.png", dpi=120)
plt.close()

# Override with 4 for interpretability (elbow + business meaning)
K_FINAL = 4
print(f"\nChosen k = {K_FINAL} (balancing silhouette score and business interpretability)")

# ─── 4. Final Clustering ─────────────────────────────────────────────────────
km_final = KMeans(n_clusters=K_FINAL, random_state=SEED, n_init=20)
df_clust["cluster"] = km_final.fit_predict(X_scaled)
print(f"Cluster sizes:\n{df_clust['cluster'].value_counts().sort_index()}")

# ─── 5. PCA Visualisation ─────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
df_clust["pca1"] = X_pca[:, 0]
df_clust["pca2"] = X_pca[:, 1]
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

CLUSTER_COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: by cluster
for cl in range(K_FINAL):
    mask = df_clust["cluster"] == cl
    axes[0].scatter(df_clust.loc[mask, "pca1"], df_clust.loc[mask, "pca2"],
                    s=5, alpha=0.4, color=CLUSTER_COLORS[cl], label=f"Cluster {cl}")
axes[0].set_title("PCA Projection — Clusters", fontweight="bold")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].legend(markerscale=5)

# Right: by default
for val, color, label in [(0, "#2ecc71", "No Default"), (1, "#e74c3c", "Default")]:
    mask = df_clust["defaulted"] == val
    axes[1].scatter(df_clust.loc[mask, "pca1"], df_clust.loc[mask, "pca2"],
                    s=5, alpha=0.3, color=color, label=label)
axes[1].set_title("PCA Projection — Default Status", fontweight="bold")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].legend(markerscale=5)

plt.tight_layout()
plt.savefig("reports/11_pca_clusters.png", dpi=120)
plt.close()

# ─── 6. Cluster Profiling ─────────────────────────────────────────────────────
profile_cols = ["monthly_income", "credit_score", "loan_amount",
                "emi_to_income", "late_payments_last_12m", "defaulted",
                "down_payment_pct", "existing_loans"]
profile = df_clust.groupby("cluster")[profile_cols].agg(["mean", "count"])
profile.columns = ["_".join(c) for c in profile.columns]
print("\n--- Cluster Profiles ---")
print(profile.round(2).T.to_string())

# Flatten for display
profile_simple = df_clust.groupby("cluster")[profile_cols].mean().round(2)
profile_simple["n_customers"] = df_clust.groupby("cluster")["defaulted"].count()
profile_simple["default_rate_pct"] = (profile_simple["defaulted"] * 100).round(1)
print("\n")
print(profile_simple.to_string())

# ─── 7. Business Names and Recommended Actions ───────────────────────────────
# Based on the profile, assign names and actions
# (These are robust heuristics; actual names are assigned from profiling)
def assign_cluster_name(row):
    income    = row["monthly_income"]
    dr        = row["default_rate_pct"]
    late      = row["late_payments_last_12m"]
    emi_ratio = row["emi_to_income"]
    dp        = row["down_payment_pct"]

    if dr > 40 or (emi_ratio > 0.40 and income < 20_000):
        return ("High-Risk Over-Leveraged",
                "Early collections follow-up at Day 15 past due. "
                "Freeze credit line increase. Mandate co-borrower for refinancing.")
    elif dr >= 10:
        return ("Vulnerable Moderate-Risk",
                "Cap LTV at 75%. Require post-dated cheques / e-NACH auto-debit. "
                "Intensive early payment reminders.")
    elif dp > 20:
        return ("Conservative Low-LTV Borrowers",
                "Offer pre-approved two-wheeler accessories financing & loyalty discounts. "
                "Fast-track processing.")
    else:
        return ("Prime Salaried Growth Segment",
                "Offer premium two-wheeler loans, personal loan cross-sells, "
                "and zero-processing fee upgrades.")

cluster_names = {}
for cl in range(K_FINAL):
    row = profile_simple.loc[cl]
    name, action = assign_cluster_name(row)
    cluster_names[cl] = {"name": name, "action": action}
    print(f"\nCluster {cl}: {name}")
    print(f"  Default rate : {row['default_rate_pct']:.1f}%")
    print(f"  Avg income   : {row['monthly_income']:,.0f}")
    print(f"  Avg credit   : {row['credit_score']:.0f}")
    print(f"  Recommended  : {action}")

profile_simple["cluster_name"] = [cluster_names[i]["name"] for i in profile_simple.index]
profile_simple["recommended_action"] = [cluster_names[i]["action"] for i in profile_simple.index]
profile_simple.to_csv("reports/cluster_profiles.csv")
print("\nCluster profiles saved -> reports/cluster_profiles.csv")

# ─── 8. Cluster Profile Heatmap ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
heat_df = profile_simple[["monthly_income", "credit_score", "emi_to_income",
                           "late_payments_last_12m", "default_rate_pct"]].copy()
heat_df.index = [f"C{i}: {cluster_names[i]['name']}" for i in heat_df.index]
# Normalize each column to 0-1 for heatmap visibility
heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
sns.heatmap(heat_norm, annot=heat_df.round(1), fmt="g", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar=False, annot_kws={"size": 9})
ax.set_title("Cluster Profile Heatmap (normalized)", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig("reports/12_cluster_heatmap.png", dpi=120)
plt.close()

# ─── 9. Test: Does cluster label improve the model? ──────────────────────────
print("\n--- Testing: Cluster Label as Feature ---")

# Map cluster to full dataset
df_full = pd.read_csv(os.path.join("data", "loans_features.csv"))
df_clust_map = df_clust[["customer_id", "cluster"]].set_index("customer_id")
df_full = df_full.join(df_clust_map, on="customer_id", how="left")
df_full["cluster"] = df_full["cluster"].fillna(-1).astype(int)

TARGET      = "defaulted"
BASE_FEATS  = ["age", "monthly_income", "credit_score", "existing_loans",
               "loan_amount", "tenure_months", "interest_rate", "emi",
               "down_payment_pct", "late_payments_last_12m",
               "emi_to_income", "ltv_ratio"]
WITH_CLUSTER = BASE_FEATS + ["cluster"]

# Remove rows with NaN in base features
df_model = df_full.dropna(subset=BASE_FEATS + [TARGET])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for feat_set, label in [(BASE_FEATS, "Without Cluster"), (WITH_CLUSTER, "With Cluster")]:
    X = df_model[feat_set].fillna(0)
    y = df_model[TARGET]
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"  {label}: ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# Save KMeans model and scaler for Streamlit app
joblib.dump({
    "kmeans":          km_final,
    "scaler":          scaler,
    "cluster_features": CLUSTER_FEATURES,
    "cluster_names":   cluster_names,
    "profile":         profile_simple,
}, "models/segmentation.joblib")

# Save full dataset with cluster labels
df_full.to_csv("data/loans_with_clusters.csv", index=False)
print("\nSegmentation model saved -> models/segmentation.joblib")
print("Dataset with clusters saved -> data/loans_with_clusters.csv")
print("\nSegmentation complete. Charts in reports/")
