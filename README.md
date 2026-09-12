# M3-PULSE — Hangry Predictive Logistics & Utilisation System

**"Predict the Risk. Recover the Capacity. Call External Capacity Only When Necessary."**

Interactive prototype for **Logistics Case Competition (LCC) 2026** — case study **Hangry**
(F&B distribution, Jakarta). Revised to a **hybrid architecture**:

```
Excel Dataset
     |
     +----------------> Live KPI / Root Cause / Capacity Analytics (recomputed every run)
     |
     +----------------> Feature engineering IDENTICAL to Colab
                                |
                                v
                      Preprocessor from Colab (models/preprocessor.pkl)
                                |
                                v
                      Model from Colab:
                        - Logistic Regression -> OPERATIONAL RISK SCORER
                        - XGBoost              -> INTERPRETABILITY / SHAP only
                                |
                                v
                      OTD Risk Prediction -> SHAP -> Decision Support -> Streamlit UI
```

**This app never trains a model.** All predictive artifacts are produced once by the Colab
notebook `LCC2026_Hangry_Analytical_Engine.ipynb` and simply loaded here with `joblib`/`json`.

## Project Structure

```
LCC2026_HANGRY/
│
├── app.py
├── requirements.txt
├── README.md
├── delivery_log_case_dataset_LCC2026.xlsx   # (or place under data/ — both paths are checked)
│
├── models/                                   # exported by the Colab notebook
│   ├── logistic_regression_otd_model.pkl     # OPERATIONAL SCORER
│   ├── xgboost_otd_model.pkl                 # INTERPRETABILITY (SHAP) only
│   ├── preprocessor.pkl
│   ├── feature_columns.pkl
│   └── model_metadata.json
│
└── outputs/                                   # exported by the Colab notebook (reference)
    ├── risk_predictions.csv, high_risk_trips.csv
    ├── kpi_summary.csv, monthly_kpi.csv
    ├── root_cause_scorecard.csv, capacity_opportunities.csv
    ├── decision_results.csv, model_metrics.csv
    ├── feature_importance.csv, shap_global.csv
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `models/*.pkl` are missing, every predictive page shows a clear on-screen error telling you
to run the Colab notebook first — it will **not** silently retrain anything.

## What Changed From the Previous Version (Audit Summary)

| Component | Before | Now | Why |
|---|---|---|---|
| Model training | Trained LR + XGBoost/RF **live in the browser** every session | **Loads** `logistic_regression_otd_model.pkl` + `xgboost_otd_model.pkl` + `preprocessor.pkl` via joblib | Colab is now the single source of truth for the model |
| Feature set | 6 features (no Vehicle Capacity / Load Carried / Trip Cost / fleet_hist_late_rate) | 10 features — synchronized with `feature_columns.pkl` / `model_metadata.json` | Prevents schema mismatch against the Colab-fitted preprocessor |
| Operational risk scorer | Whichever model "won" a live ROC-AUC comparison | **Logistic Regression**, explicitly — Colab's hold-out showed LR recall 0.708 vs XGBoost 0.111 | Business objective is recall (catch real late deliveries); this is Colab's own evidence-based decision, not a Streamlit assumption |
| SHAP / explainability | Ran on whichever model won | Always runs on the **XGBoost artifact** specifically, via TreeExplainer | XGBoost is kept purely for interpretability, matching Colab's final architecture |
| Model & Data Governance page | Read from live training session state | Reads directly from `models/model_metadata.json` | Single source of truth; works even without visiting Risk Monitor first |
| KPI / Root Cause / Capacity / Decision pages | Live-computed from Excel | **Unchanged** — still live-computed from Excel | Opsi B: operational KPIs should react to data changes without re-running Colab |
| Error handling | None — missing files would crash | Friendly "Model artifact not found" / "Feature schema mismatch" messages | Per spec section V |
| SHAP -> Operational Insight | Not present | Trip Risk Detail page now shows Factor -> Interpretation -> Potential Action table | Per spec section J |

Everything else — layout, sidebar, navigation, color theme, KPI cards, Decision Gate rules,
Decision Trace, Scenario Simulator, Alert Center, Demo Mode, Management/Planner toggle — is
**unchanged** from the previous version.

## Data Integrity Rules Enforced in Code

- `load_model_artifacts()` never trains — it loads or shows an error.
- `engineer_features_for_scoring()` reproduces the Colab notebook's feature engineering
  exactly (same historical-rate expanding-average logic, same leakage exclusions).
- Historical driver/route/fleet late-rate features use **expanding averages computed only from
  legs strictly before the current leg's timestamp** — no future information leaks in.
- `Actual Arrival`, `Delta (min)`, `Reason Code`, `On-Time Flag` are never used as predictors.
- SHAP output is captioned as *"Model indicates this feature contributed to the predicted
  risk"* — never phrased as a causal claim.
- Capacity opportunities are labeled **"Potential Capacity Recovery"**, never "Confirmed
  Consolidation".
- Cost comparisons use actual Rupiah figures where the data supports it, and are labeled
  **"Qualitative assessment"** everywhere else — no invented monetary savings.

## Known Limitation (carried over honestly from Colab)

The Colab hold-out validation covered a single month (~339 legs) — `model_metadata.json`
states this explicitly, and the Model & Data Governance page surfaces it verbatim. Treat model
metrics as directionally indicative of feasibility, not production-grade validation.
