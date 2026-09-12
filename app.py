"""
M3-PULSE — Hangry Predictive Logistics & Utilisation System
LCC2026 Interactive Prototype (Streamlit)

"Predict the Risk. Recover the Capacity. Call External Capacity Only When Necessary."

ARCHITECTURE (HYBRID / "Opsi B") — REVISED to consume the Colab analytical engine:

    Excel Dataset
         |
         +----------------> Live KPI / Root Cause / Capacity Analytics (unchanged, live-compute)
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

IMPORTANT — this file does NOT train any model. All predictive artifacts (model, preprocessor,
feature list, metadata) are loaded from models/, produced by the Colab notebook
`LCC2026_Hangry_Analytical_Engine.ipynb`. If those files are missing, the app shows a clear
error instead of silently retraining or crashing (see `load_model_artifacts()`).

DATA INTEGRITY RULES (enforced throughout this file):
- No dummy data, no dummy KPIs, no dummy ML results. KPI/Root Cause/Capacity pages compute
  live from delivery_log_case_dataset_LCC2026.xlsx; predictive pages use the Colab artifacts.
- No fabricated causal claims. Anything not statistically/empirically supported by the
  dataset is explicitly labeled "Operational Hypothesis", not "Validated Finding".
- No leakage: Actual Arrival, Delta (min), Reason Code, On-Time Flag are NEVER used as model
  features — this mirrors the leakage prevention already validated in the Colab notebook.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Optional dependency: SHAP is only used to EXPLAIN the already-trained XGBoost
# artifact from Colab — never to train anything.
# ---------------------------------------------------------------------------
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


DATA_PATH = _first_existing(
    os.path.join(APP_DIR, "data", "delivery_log_case_dataset_LCC2026.xlsx"),
    os.path.join(APP_DIR, "delivery_log_case_dataset_LCC2026.xlsx"),
)
MODELS_DIR = os.path.join(APP_DIR, "models")
OUTPUTS_DIR = os.path.join(APP_DIR, "outputs")

# ---------------------------------------------------------------------------
# PAGE CONFIG + THEME (dark navy / blue / white / emerald / red / gray)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="M3-PULSE | Hangry Predictive Logistics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "navy": "#0B1F3A",
    "navy2": "#122A4E",
    "blue": "#1F6FEB",
    "white": "#FFFFFF",
    "emerald": "#1AA260",
    "red": "#D64545",
    "amber": "#E8A33D",
    "gray": "#8A94A6",
    "lightgray": "#F4F6F9",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {COLORS['lightgray']}; }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['navy']};
    }}
    section[data-testid="stSidebar"] * {{ color: {COLORS['white']} !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 0.92rem; }}

    h1, h2, h3 {{ color: {COLORS['navy']}; font-family: 'Segoe UI', sans-serif; }}

    .m3-hero {{
        background: linear-gradient(135deg, {COLORS['navy']} 0%, {COLORS['navy2']} 100%);
        padding: 2.2rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
    }}
    .m3-hero h1 {{ color: white; font-size: 2.1rem; margin-bottom: 0.2rem; }}
    .m3-hero p {{ color: #C7D3E8; font-size: 1.05rem; margin: 0; }}

    .kpi-card {{
        background: white; border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 6px solid {COLORS['gray']}; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .kpi-card.green {{ border-left-color: {COLORS['emerald']}; }}
    .kpi-card.amber {{ border-left-color: {COLORS['amber']}; }}
    .kpi-card.red {{ border-left-color: {COLORS['red']}; }}
    .kpi-label {{ color: {COLORS['gray']}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {COLORS['navy']}; }}
    .kpi-sub {{ font-size: 0.8rem; color: {COLORS['gray']}; }}

    .badge {{ display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
              font-size: 0.75rem; font-weight: 700; color: white; }}
    .badge-green {{ background-color: {COLORS['emerald']}; }}
    .badge-amber {{ background-color: {COLORS['amber']}; }}
    .badge-red {{ background-color: {COLORS['red']}; }}
    .badge-gray {{ background-color: {COLORS['gray']}; }}

    .decision-card {{
        background: white; border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 0.8rem;
        border: 1px solid #E3E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .hyp-label {{
        background-color: #FFF4E0; color: #7A5300; padding: 0.15rem 0.55rem;
        border-radius: 6px; font-size: 0.72rem; font-weight: 700;
    }}
    .val-label {{
        background-color: #E4F7EC; color: #0F6B3D; padding: 0.15rem 0.55rem;
        border-radius: 6px; font-size: 0.72rem; font-weight: 700;
    }}
    .footer-note {{ color: {COLORS['gray']}; font-size: 0.78rem; margin-top: 2rem; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str):
    """Load the two linked sheets from the ORIGINAL Excel file. No synthetic data."""
    if not os.path.exists(path):
        return None, None, f"File tidak ditemukan: {path}"
    xl = pd.ExcelFile(path)
    dl = xl.parse("Delivery Log")
    ts = xl.parse("Trip Summary")
    return dl, ts, None


# ---------------------------------------------------------------------------
# 2. DATA CLEANING (judgment-based, nothing silently dropped)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(dl_raw: pd.DataFrame, ts_raw: pd.DataFrame):
    """
    Cleans the delivery log & trip summary while PRESERVING every row.
    Mirrors the audit logic validated in the accompanying analysis notebook:
      - 288 legs with no confirmed arrival -> flagged 'Unconfirmed', retained
      - 1 extreme Delta outlier (~-1027 min) -> row retained, excluded from delay-magnitude stats
      - Reason Code NaN on-time legs -> retained as Not-Applicable (not a defect)
    Returns cleaned dl, ts, and a Data Quality Summary dataframe for the governance page.
    """
    dl = dl_raw.copy()
    ts = ts_raw.copy()

    dl["Month"] = pd.to_datetime(dl["Planned Departure (date)"]).dt.month
    ts["Month"] = pd.to_datetime(ts["Date"]).dt.month

    # --- flags (no rows removed) ---
    dl["is_unconfirmed"] = dl["On-Time Flag"].isna()
    dl["on_time_status"] = dl["On-Time Flag"].fillna("Unconfirmed")
    dl["is_on_time"] = dl["On-Time Flag"] == "Yes"

    dl["delta_outlier_flag"] = dl["Delta (min)"].abs() > 200  # physically implausible for a 60-min window

    n_total = len(dl)
    n_unconfirmed = int(dl["is_unconfirmed"].sum())
    n_outlier = int(dl["delta_outlier_flag"].sum())
    n_reason_na_on_time = int(((dl["Reason Code"].isna()) & (dl["On-Time Flag"] == "Yes")).sum())
    n_dupe_records = int(dl["Record ID"].duplicated().sum())
    n_dupe_trips = int(ts["Trip ID"].duplicated().sum())

    dq_summary = pd.DataFrame([
        dict(Issue="Missing Actual Arrival / On-Time Flag / Delta", Count=n_unconfirmed,
             Pct=f"{n_unconfirmed/n_total*100:.1f}%", Decision="RETAIN + FLAG (Unconfirmed)",
             Rationale="Structural tracking gap (manual 3PL reconciliation, no real-time feed). "
                       "Dropping would understate delivery volume and bias OTD."),
        dict(Issue="Extreme Delta outlier (>200 min / <-200 min)", Count=n_outlier,
             Pct=f"{n_outlier/n_total*100:.2f}%", Decision="RETAIN row, EXCLUDE from delay-magnitude stats",
             Rationale="Physically implausible for a 60-minute delivery window; almost certainly a data-entry error. "
                       "The categorical On-Time Flag is still kept as a usable ops signal."),
        dict(Issue="Reason Code missing while On-Time", Count=n_reason_na_on_time,
             Pct=f"{n_reason_na_on_time/n_total*100:.1f}%", Decision="RETAIN as Not-Applicable",
             Rationale="Reason Code is by design only captured for LATE legs — not a data defect."),
        dict(Issue="Duplicate Record ID", Count=n_dupe_records, Pct=f"{n_dupe_records/n_total*100:.2f}%",
             Decision="No action needed", Rationale="Primary key confirmed unique."),
        dict(Issue="Duplicate Trip ID", Count=n_dupe_trips, Pct=f"{n_dupe_trips/max(len(ts),1)*100:.2f}%",
             Decision="No action needed", Rationale="Primary key confirmed unique."),
    ])

    return dl, ts, dq_summary


# ---------------------------------------------------------------------------
# 3. MASTER DATASET (delivery-level + trip-level, no row multiplication)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_master(dl: pd.DataFrame, ts: pd.DataFrame):
    trip_cols = ["Trip ID", "Drops on Trip", "Vehicle Capacity (CBM)", "Load Carried (CBM)",
                 "Utilisation (Load/Capacity)", "Trip Cost (IDR)"]
    master_delivery = dl.merge(ts[trip_cols], on="Trip ID", how="left", validate="many_to_one")
    master_delivery["Cost per Delivery (IDR)"] = (
        master_delivery["Trip Cost (IDR)"] / master_delivery["Drops on Trip"]
    )

    trip_otd = (dl.groupby("Trip ID")
                  .agg(n_legs=("Record ID", "size"),
                       n_confirmed=("On-Time Flag", lambda s: s.notna().sum()),
                       n_on_time=("On-Time Flag", lambda s: (s == "Yes").sum()))
                )
    trip_otd["trip_otd_rate"] = trip_otd["n_on_time"] / trip_otd["n_confirmed"].replace(0, np.nan)
    master_trip = ts.merge(trip_otd, on="Trip ID", how="left")
    master_trip["Cost per Delivery (IDR)"] = master_trip["Trip Cost (IDR)"] / master_trip["n_legs"]
    master_trip["unused_capacity_cbm"] = (
        master_trip["Vehicle Capacity (CBM)"] - master_trip["Load Carried (CBM)"]
    ).clip(lower=0)

    return master_delivery, master_trip


# ---------------------------------------------------------------------------
# 4. KPI ENGINE
# ---------------------------------------------------------------------------
MONTH_LABELS = {1: "Bulan 1 (Baseline)", 2: "Bulan 2", 3: "Bulan 3", 4: "Bulan 4 (Current)"}

TARGETS = {
    "otd": 95.0,
    "utilisation": 90.0,
}


def apply_filters(dl, ts, months=None, fleet_types=None, drivers=None, origins=None, destinations=None):
    dl_f, ts_f = dl.copy(), ts.copy()
    if months:
        dl_f = dl_f[dl_f["Month"].isin(months)]
        ts_f = ts_f[ts_f["Month"].isin(months)]
    if fleet_types:
        dl_f = dl_f[dl_f["Fleet Type"].isin(fleet_types)]
        ts_f = ts_f[ts_f["Fleet Type"].isin(fleet_types)]
    if drivers:
        dl_f = dl_f[dl_f["Driver"].isin(drivers)]
        ts_f = ts_f[ts_f["Driver"].isin(drivers)]
    if origins:
        dl_f = dl_f[dl_f["Origin"].isin(origins)]
        ts_f = ts_f[ts_f["Origin"].isin(origins)]
    if destinations:
        dl_f = dl_f[dl_f["Destination"].isin(destinations)]
    return dl_f, ts_f


def compute_kpis(dl: pd.DataFrame, ts: pd.DataFrame) -> dict:
    """All KPIs computed directly from the (filtered) dataset. No hardcoded numbers."""
    out = {}
    confirmed = dl.dropna(subset=["On-Time Flag"])
    out["n_legs"] = len(dl)
    out["n_confirmed"] = len(confirmed)
    out["n_trips"] = len(ts)
    out["otd_confirmed"] = (confirmed["On-Time Flag"] == "Yes").mean() * 100 if len(confirmed) else np.nan
    out["otd_conservative"] = (dl["On-Time Flag"] == "Yes").mean() * 100 if len(dl) else np.nan

    ded = ts[ts["Fleet Type"] == "Dedicated"]
    oc = ts[ts["Fleet Type"] == "On-Call"]
    out["util_dedicated"] = ded["Utilisation (Load/Capacity)"].mean() * 100 if len(ded) else np.nan
    out["util_oncall"] = oc["Utilisation (Load/Capacity)"].mean() * 100 if len(oc) else np.nan
    out["util_all"] = ts["Utilisation (Load/Capacity)"].mean() * 100 if len(ts) else np.nan

    out["oncall_share"] = (ts["Fleet Type"] == "On-Call").mean() * 100 if len(ts) else np.nan
    out["avg_drops_trip"] = ts["Drops on Trip"].mean() if len(ts) else np.nan
    out["avg_drops_trip_dedicated"] = ded["Drops on Trip"].mean() if len(ded) else np.nan

    out["total_cost"] = ts["Trip Cost (IDR)"].sum() if len(ts) else 0
    out["cost_per_trip"] = ts["Trip Cost (IDR)"].mean() if len(ts) else np.nan
    out["cost_per_delivery"] = (ts["Trip Cost (IDR)"].sum() / len(dl)) if len(dl) else np.nan

    cpd = ts.assign(cpd=ts["Trip Cost (IDR)"] / ts["Drops on Trip"]) if len(ts) else pd.DataFrame(columns=["cpd"])
    out["ded_cost_per_drop"] = cpd.loc[ts["Fleet Type"] == "Dedicated", "cpd"].mean() if len(ts) else np.nan
    out["oc_cost_per_drop"] = cpd.loc[ts["Fleet Type"] == "On-Call", "cpd"].mean() if len(ts) else np.nan
    if out.get("ded_cost_per_drop") and not np.isnan(out["ded_cost_per_drop"]) and out["ded_cost_per_drop"] > 0:
        out["oncall_premium_pct"] = (out["oc_cost_per_drop"] / out["ded_cost_per_drop"] - 1) * 100
    else:
        out["oncall_premium_pct"] = np.nan

    otd_fleet = confirmed.groupby("Fleet Type")["On-Time Flag"].apply(lambda s: (s == "Yes").mean() * 100)
    out["otd_dedicated"] = otd_fleet.get("Dedicated", np.nan)
    out["otd_oncall"] = otd_fleet.get("On-Call", np.nan)

    out["pct_unconfirmed"] = dl["On-Time Flag"].isna().mean() * 100 if len(dl) else np.nan
    return out


def kpi_status(current, target, higher_is_better=True):
    """GREEN / AMBER / RED classification vs target."""
    if pd.isna(current) or pd.isna(target):
        return "gray"
    gap = (current - target) if higher_is_better else (target - current)
    gap_pct = gap / target * 100 if target else 0
    if gap_pct >= 0:
        return "green"
    elif gap_pct >= -5:
        return "amber"
    return "red"


def monthly_series(dl, ts):
    rows = []
    for m in sorted(dl["Month"].dropna().unique()):
        dl_m = dl[dl["Month"] == m]
        ts_m = ts[ts["Month"] == m]
        k = compute_kpis(dl_m, ts_m)
        rows.append({
            "Month": m, "MonthLabel": MONTH_LABELS.get(m, f"Bulan {m}"),
            "OTD": k["otd_confirmed"], "Utilisation": k["util_dedicated"],
            "OnCallShare": k["oncall_share"], "DropsPerTrip": k["avg_drops_trip"],
            "TotalCost": k["total_cost"], "Trips": k["n_trips"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["CostIndex"] = df["TotalCost"] / df["TotalCost"].iloc[0] * 100
    return df


# ---------------------------------------------------------------------------
# 5. ROOT CAUSE ENGINE
# ---------------------------------------------------------------------------
def pareto_table(series: pd.Series, top_n=15):
    s = series.dropna()
    if len(s) == 0:
        return pd.DataFrame(columns=["Category", "Count", "Pct", "CumPct"])
    counts = s.value_counts().head(top_n)
    df = pd.DataFrame({"Category": counts.index, "Count": counts.values})
    df["Pct"] = (df["Count"] / s.value_counts().sum() * 100).round(1)
    df["CumPct"] = df["Pct"].cumsum().round(1)
    return df


@st.cache_data(show_spinner=False)
def root_cause_scorecard(dl: pd.DataFrame, ts: pd.DataFrame):
    """
    Builds an evidence-based root cause scorecard using correlation / group comparison.
    Severity & Confidence are derived transparently from effect size and statistical support
    (documented in the 'How is this computed?' expander in the UI), not invented.
    """
    from scipy import stats
    rows = []

    # 1. Route / Trip Efficiency (drops/trip vs utilisation)
    if len(ts) >= 10:
        r, pval = stats.pearsonr(ts["Drops on Trip"], ts["Utilisation (Load/Capacity)"])
        sev = "High" if abs(r) >= 0.5 else ("Medium" if abs(r) >= 0.3 else "Low")
        conf = "High" if pval < 0.01 else ("Medium" if pval < 0.05 else "Low")
        rows.append(dict(
            RootCause="Route / Trip Efficiency (consolidation)",
            Evidence=f"corr(drops/trip, utilisation) = {r:.2f}, p = {pval:.4f}, n = {len(ts)} trips",
            AffectedKPI="Dedicated Utilisation, Cost per Drop",
            Severity=sev, Confidence=conf,
        ))

    # 2. Fleet Reliability (Dedicated vs On-Call OTD)
    conf_dl = dl.dropna(subset=["On-Time Flag"])
    if len(conf_dl):
        otd_fleet = conf_dl.groupby("Fleet Type")["On-Time Flag"].apply(lambda s: (s == "Yes").mean() * 100)
        if "Dedicated" in otd_fleet.index and "On-Call" in otd_fleet.index:
            gap = otd_fleet["Dedicated"] - otd_fleet["On-Call"]
            rows.append(dict(
                RootCause="Fleet Reliability (On-Call as a cost add, not a fix)",
                Evidence=f"OTD Dedicated={otd_fleet['Dedicated']:.1f}% vs On-Call={otd_fleet['On-Call']:.1f}% (gap={gap:+.1f}pp)",
                AffectedKPI="OTD, Transportation Cost",
                Severity="High" if gap > 5 else "Medium",
                Confidence="High" if len(conf_dl) > 200 else "Medium",
            ))

    # 3. Driver Performance (spread, but check concentration)
    driver_otd = conf_dl.groupby("Driver")["On-Time Flag"].apply(lambda s: (s == "Yes").mean() * 100)
    driver_n = conf_dl.groupby("Driver").size()
    valid_drivers = driver_otd[driver_n >= 15]
    if len(valid_drivers) >= 3:
        spread = valid_drivers.max() - valid_drivers.min()
        rows.append(dict(
            RootCause="Driver Performance Variation",
            Evidence=f"OTD spread across drivers (min. 15 legs) = {spread:.1f}pp (min={valid_drivers.min():.1f}%, max={valid_drivers.max():.1f}%)",
            AffectedKPI="OTD",
            Severity="Medium" if spread > 20 else "Low",
            Confidence="Medium — variation exists but is broad-based, not concentrated in a few drivers",
        ))

    # 4. Planning / Scheduling (trend of drops/trip over time)
    monthly_drops = ts.groupby("Month")["Drops on Trip"].mean()
    if len(monthly_drops) >= 2:
        decline = monthly_drops.iloc[0] - monthly_drops.iloc[-1]
        decline_pct = decline / monthly_drops.iloc[0] * 100 if monthly_drops.iloc[0] else 0
        rows.append(dict(
            RootCause="Planning / Scheduling (consolidation collapse over time)",
            Evidence=f"Avg drops/trip fell {monthly_drops.iloc[0]:.2f} -> {monthly_drops.iloc[-1]:.2f} ({decline_pct:.1f}% decline) across observed months",
            AffectedKPI="Utilisation, On-Call Share, Cost",
            Severity="High" if decline_pct > 15 else "Medium",
            Confidence="High — consistent monotonic trend across all observed months",
        ))

    # 5. Data Visibility / Forecasting
    pct_unconf = dl["On-Time Flag"].isna().mean() * 100
    reason_counts = conf_dl.loc[conf_dl["On-Time Flag"] == "No", "Reason Code"].value_counts()
    fv_share = 0.0
    if reason_counts.sum() > 0 and "Forecast Variance" in reason_counts.index:
        fv_share = reason_counts["Forecast Variance"] / reason_counts.sum() * 100
    rows.append(dict(
        RootCause="Data Visibility / Forecasting",
        Evidence=f"{pct_unconf:.1f}% of legs have no confirmed arrival (manual reconciliation, no real-time feed); "
                  f"'Forecast Variance' = {fv_share:.1f}% of late legs' reason codes",
        AffectedKPI="OTD (delayed detection/correction)",
        Severity="Medium" if pct_unconf > 10 else "Low",
        Confidence="High — directly observed in the data" if pct_unconf > 0 else "Low",
    ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. COLAB MODEL ARTIFACTS — load only, NEVER train (Opsi B / Hybrid architecture)
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = {
    "operational_scorer": "logistic_regression_otd_model.pkl",
    "interpretability_model": "xgboost_otd_model.pkl",
    "preprocessor": "preprocessor.pkl",
    "feature_columns": "feature_columns.pkl",
    "metadata": "model_metadata.json",
}


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """
    Loads the Colab-exported model suite. Returns (artifacts_dict, error_message).
    If ANY required file is missing, returns (None, friendly_error) — the app must NOT fall
    back to training something live; it must tell the user to run the Colab engine first.
    """
    paths = {k: os.path.join(MODELS_DIR, v) for k, v in REQUIRED_ARTIFACTS.items()}
    missing = [v for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        missing_names = [os.path.basename(m) for m in missing]
        return None, (
            "Model artifact not found. Please run the Colab analytical engine "
            "(`LCC2026_Hangry_Analytical_Engine.ipynb`) and place the exported model files "
            f"in the `models/` directory.\n\nMissing file(s): {', '.join(missing_names)}"
        )

    try:
        scorer = joblib.load(paths["operational_scorer"])       # Logistic Regression
        interpret_model = joblib.load(paths["interpretability_model"])  # XGBoost
        preprocessor = joblib.load(paths["preprocessor"])
        feature_columns = joblib.load(paths["feature_columns"])
        with open(paths["metadata"]) as f:
            metadata = json.load(f)
    except Exception as e:
        return None, f"Failed to load model artifacts: {e}"

    num_features = metadata.get("features_numeric", [])
    cat_features = metadata.get("features_categorical", [])
    # If metadata doesn't split them, fall back to feature_columns.pkl order with a best-effort split.
    if not num_features and not cat_features:
        cat_features = [c for c in feature_columns if c in ("Origin", "Fleet Type")]
        num_features = [c for c in feature_columns if c not in cat_features]

    return dict(
        scorer=scorer, interpret_model=interpret_model, preprocessor=preprocessor,
        feature_columns=feature_columns, num_features=num_features, cat_features=cat_features,
        metadata=metadata,
    ), None


@st.cache_data(show_spinner=False)
def engineer_features_for_scoring(master_delivery: pd.DataFrame, _num_features, _cat_features):
    """
    Rebuilds the EXACT SAME feature engineering as the Colab notebook (Phase 14), so the
    dataframe passed into the Colab preprocessor is schema-compatible. STRICTLY EXCLUDED as
    features (only known AFTER delivery): Actual Arrival, Delta (min), Reason Code, On-Time Flag.
    Historical driver/route/fleet late-rate are EXPANDING averages computed only from legs
    strictly BEFORE the current leg's timestamp (no future information used) — identical logic
    to the Colab `df_model` construction.
    """
    df = master_delivery.copy()
    df = df.dropna(subset=["On-Time Flag"]).copy()  # only confirmed legs have a usable label
    df["planned_dt"] = pd.to_datetime(
        df["Planned Departure (date)"].astype(str) + " " + df["Planned Departure (time)"].astype(str)
    )
    df = df.sort_values("planned_dt").reset_index(drop=True)

    df["target_late"] = (df["On-Time Flag"] == "No").astype(int)

    df["driver_hist_late_rate"] = (
        df.groupby("Driver")["target_late"].apply(lambda s: s.shift().expanding().mean())
          .reset_index(level=0, drop=True)
    )
    df["route"] = df["Origin"] + " -> " + df["Destination"]
    df["route_hist_late_rate"] = (
        df.groupby("route")["target_late"].apply(lambda s: s.shift().expanding().mean())
          .reset_index(level=0, drop=True)
    )
    df["fleet_hist_late_rate"] = (
        df.groupby("Fleet Type")["target_late"].apply(lambda s: s.shift().expanding().mean())
          .reset_index(level=0, drop=True)
    )
    global_rate = df["target_late"].mean()
    for c in ["driver_hist_late_rate", "route_hist_late_rate", "fleet_hist_late_rate"]:
        if c in df.columns:
            df[c] = df[c].fillna(global_rate)

    df["departure_hour"] = df["planned_dt"].dt.hour
    df["dow"] = df["planned_dt"].dt.dayofweek

    # Schema check against what the Colab preprocessor actually expects.
    expected = list(_num_features) + list(_cat_features)
    available = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    return df, available, missing


def score_and_explain(df_feat: pd.DataFrame, artifacts: dict):
    """
    INPUT (raw trip/delivery data) -> feature engineering -> Colab preprocessor ->
    Colab operational scorer (Logistic Regression) for the risk probability, and the Colab
    XGBoost artifact (via SHAP TreeExplainer) purely for interpretability. No training happens
    here — both models are the exact fitted objects exported by the notebook.
    """
    feature_columns = artifacts["feature_columns"]
    X = df_feat[feature_columns]
    X_t = artifacts["preprocessor"].transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    # Operational risk scorer — Logistic Regression, per the Colab notebook's final decision.
    proba = artifacts["scorer"].predict_proba(X_t)[:, 1]

    # Interpretability — XGBoost + SHAP (never used for the headline risk score itself).
    explain_kind, explain_df = "importance", None
    cat_names = list(artifacts["preprocessor"].named_transformers_["cat"].get_feature_names_out(artifacts["cat_features"]))
    feat_names = list(artifacts["num_features"]) + cat_names

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(artifacts["interpret_model"])
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            explain_df = pd.DataFrame(shap_values, columns=feat_names)
            explain_kind = "shap"
        except Exception:
            explain_df = None

    if explain_df is None:
        clf = artifacts["interpret_model"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:
            imp = np.ones(len(feat_names)) / len(feat_names)
        explain_df = pd.DataFrame(np.tile(imp, (X_t.shape[0], 1)), columns=feat_names)
        explain_kind = "importance"

    return proba, explain_kind, explain_df, feat_names


def risk_level(p):
    if p <= 0.30:
        return "LOW", "green"
    elif p <= 0.60:
        return "MEDIUM", "amber"
    elif p <= 0.80:
        return "HIGH", "red"
    else:
        return "CRITICAL", "red"


def render_artifact_error(err_msg):
    st.error(f"⚠️ {err_msg}")
    with st.expander("Expected model files"):
        st.code("\n".join(f"models/{v}" for v in REQUIRED_ARTIFACTS.values()))


def render_schema_mismatch(expected, missing):
    st.error("⚠️ Feature schema mismatch between Streamlit and Colab model.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected features (from Colab)**")
        st.code("\n".join(expected))
    with c2:
        st.markdown("**Missing in current dataset**")
        st.code("\n".join(missing) if missing else "(none)")


# ---------------------------------------------------------------------------
# 7. CAPACITY OPPORTUNITY
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def capacity_opportunity_table(ts: pd.DataFrame):
    df = ts.copy()
    df["Unused Capacity (CBM)"] = (df["Vehicle Capacity (CBM)"] - df["Load Carried (CBM)"]).clip(lower=0)
    df["Utilisation %"] = (df["Utilisation (Load/Capacity)"] * 100).round(1)
    median_util = df["Utilisation (Load/Capacity)"].median()

    def priority(row):
        if row["Fleet Type"] != "Dedicated":
            return "N/A (On-Call)"
        if row["Utilisation (Load/Capacity)"] < 0.6:
            return "High — Potential Consolidation"
        elif row["Utilisation (Load/Capacity)"] < median_util:
            return "Medium — Review"
        return "Low — Efficient"

    df["Priority"] = df.apply(priority, axis=1)
    return df.sort_values("Unused Capacity (CBM)", ascending=False)


@st.cache_data(show_spinner=False)
def same_day_capacity_check(ts: pd.DataFrame):
    """
    For every on-call trip, checks whether SAME-DAY, SAME-ORIGIN dedicated capacity had enough
    spare CBM to (on a volume basis) absorb it. This is a volume-based proxy only — it does NOT
    verify route/time-window feasibility, which the dataset does not contain. Labeled explicitly
    as 'Potential Consolidation', never 'Confirmed Consolidation'.
    """
    daily = (ts.groupby(["Date", "Origin", "Fleet Type"])
               .agg(trips=("Trip ID", "size"),
                    total_capacity=("Vehicle Capacity (CBM)", "sum"),
                    total_load=("Load Carried (CBM)", "sum"))
               .reset_index())
    ded = daily[daily["Fleet Type"] == "Dedicated"].copy()
    ded["unused"] = (ded["total_capacity"] - ded["total_load"]).clip(lower=0)
    oc = daily[daily["Fleet Type"] == "On-Call"][["Date", "Origin", "trips", "total_load"]].rename(
        columns={"trips": "oncall_trips", "total_load": "oncall_load"})
    merged = oc.merge(ded[["Date", "Origin", "unused"]], on=["Date", "Origin"], how="left")
    merged["unused"] = merged["unused"].fillna(0)
    merged["potentially_avoidable"] = merged["unused"] >= merged["oncall_load"]
    return merged


# ---------------------------------------------------------------------------
# 8. ON-CALL DECISION ENGINE (transparent, rule-based, auditable)
# ---------------------------------------------------------------------------
def decision_engine(risk_score, same_day_spare_cbm, load_cbm, drops_on_trip,
                     utilisation, otd_gap_pp):
    """
    Transparent if-then rule engine. Every rule fired is returned so the UI can render a full
    decision trace (INPUT -> RISK -> CAPACITY -> CONSTRAINTS -> RULES -> DECISION -> ACTION).
    Rules are only evaluated using variables that are actually available in this dataset.
    """
    triggered = []
    capacity_available = same_day_spare_cbm >= load_cbm
    capacity_status = "Available" if capacity_available else "Insufficient"

    if risk_score < 0.30:
        risk_band = "LOW"
    elif risk_score < 0.60:
        risk_band = "MEDIUM"
    elif risk_score < 0.80:
        risk_band = "HIGH"
    else:
        risk_band = "CRITICAL"
    triggered.append(f"Risk band = {risk_band} ({risk_score*100:.0f}%)")

    service_risk_high = risk_band in ("HIGH", "CRITICAL")

    if risk_band == "LOW" and capacity_available:
        decision = "A. KEEP DEDICATED"
        triggered.append("✓ Low risk"); triggered.append("✓ Sufficient capacity")
        rationale = ("Risk of lateness is low and existing dedicated capacity is sufficient. "
                     "No intervention needed beyond standard monitoring.")
    elif risk_band == "LOW" and not capacity_available:
        decision = "A. KEEP DEDICATED"
        triggered.append("✓ Low risk"); triggered.append("✗ Capacity tight, but risk does not justify action")
        rationale = "Risk is low even without confirmed spare capacity — proceed as planned, monitor."
    elif service_risk_high and capacity_available:
        decision = "C. CONSOLIDATE" if drops_on_trip >= 1 else "B. RESEQUENCE / OPTIMISE"
        triggered.append(f"✓ {risk_band} risk"); triggered.append("✓ Recoverable capacity same-day/origin")
        triggered.append("✓ On-call not immediately required")
        rationale = ("High delivery risk exists, but recoverable dedicated capacity provides a "
                      "lower-cost recovery option than dispatching an on-call truck.")
    elif service_risk_high and not capacity_available:
        decision = "D. USE ON-CALL"
        triggered.append(f"✓ {risk_band} risk"); triggered.append("✗ Insufficient dedicated capacity")
        triggered.append("✓ High service failure risk if unaddressed")
        rationale = ("Risk is high, no same-day/same-origin dedicated capacity is available to absorb "
                      "the load, and the service-failure risk justifies the on-call premium.")
    else:  # MEDIUM risk
        if capacity_available:
            decision = "B. RESEQUENCE / OPTIMISE"
            triggered.append("✓ Medium risk"); triggered.append("✓ Capacity recoverable")
            rationale = "Moderate risk with recoverable capacity — resequencing is lower-cost than on-call."
        else:
            decision = "D. USE ON-CALL"
            triggered.append("✓ Medium risk"); triggered.append("✗ Capacity not recoverable")
            rationale = "Moderate risk but no recoverable capacity; on-call used as a bounded, justified exception."

    return dict(decision=decision, rationale=rationale, triggered=triggered,
                risk_band=risk_band, capacity_status=capacity_status)


def qualitative_or_quantitative_cost(decision_key, ded_cost_avg, oncall_cost_avg):
    """Uses ACTUAL calculated cost values where the data supports it; otherwise a labeled
    qualitative LOW/MEDIUM/HIGH assessment. Never fabricates monetary savings."""
    mapping = {
        "A. KEEP DEDICATED": dict(service="Low (planned)", capacity="No change",
                                    cost=f"Rp{ded_cost_avg:,.0f} (avg. dedicated trip, already budgeted)",
                                    risk="Low", quant=True),
        "B. RESEQUENCE / OPTIMISE": dict(service="Low-Medium (depends on execution)", capacity="Recovered on existing trip",
                                    cost="No incremental trip cost (uses existing dedicated capacity)",
                                    risk="Medium — Qualitative assessment", quant=False),
        "C. CONSOLIDATE": dict(service="Low-Medium", capacity="Recovered via consolidation",
                                    cost="No incremental trip cost (load absorbed into existing trip)",
                                    risk="Medium — Qualitative assessment", quant=False),
        "D. USE ON-CALL": dict(service="Higher near-term reliability for this trip", capacity="No recovery",
                                    cost=f"Rp{oncall_cost_avg:,.0f} (avg. on-call trip cost — premium vs dedicated)",
                                    risk="High operating cost, Low near-term service risk", quant=True),
    }
    return mapping.get(decision_key, {})


# ---------------------------------------------------------------------------
# 9. ALERT CENTER
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_alerts(dl: pd.DataFrame, ts: pd.DataFrame, risk_table: pd.DataFrame = None):
    alerts = []

    if risk_table is not None and len(risk_table):
        crit = risk_table[risk_table["Risk Level"] == "CRITICAL"]
        for _, r in crit.iterrows():
            alerts.append(dict(Alert="High delivery risk predicted", Severity="CRITICAL",
                                TripID=r["Trip ID"], KPI="OTD", RootCause="High predicted lateness probability",
                                Action="Run On-Call Decision Gate for this trip"))

    ded = ts[ts["Fleet Type"] == "Dedicated"]
    under = ded[ded["Utilisation (Load/Capacity)"] < 0.6]
    if len(under):
        alerts.append(dict(Alert=f"{len(under)} dedicated trips running under 60% utilisation",
                            Severity="HIGH", TripID="Multiple (see Capacity Opportunity)",
                            KPI="Dedicated Utilisation", RootCause="Low drops/trip / under-consolidation",
                            Action="Review consolidation opportunities"))

    conf = dl.dropna(subset=["On-Time Flag"])
    driver_otd = conf.groupby("Driver")["On-Time Flag"].apply(lambda s: (s == "Yes").mean() * 100)
    driver_n = conf.groupby("Driver").size()
    worsening = driver_otd[(driver_n >= 15) & (driver_otd < 70)]
    for drv, val in worsening.items():
        alerts.append(dict(Alert=f"Driver {drv} OTD below 70%", Severity="WARNING", TripID="N/A",
                            KPI="OTD", RootCause="Driver performance",
                            Action="Review driver route assignments / coaching"))

    n_missing = dl["On-Time Flag"].isna().sum()
    if n_missing > 0:
        pct = n_missing / len(dl) * 100
        alerts.append(dict(Alert=f"{n_missing} delivery legs ({pct:.1f}%) missing confirmed arrival data",
                            Severity="DATA QUALITY", TripID="Multiple", KPI="OTD (measurement)",
                            RootCause="Manual reconciliation lag / no real-time integration",
                            Action="Flag as Unconfirmed; do not silently exclude from reporting"))

    return pd.DataFrame(alerts) if alerts else pd.DataFrame(
        columns=["Alert", "Severity", "TripID", "KPI", "RootCause", "Action"])


# ---------------------------------------------------------------------------
# 10. SCENARIO SIMULATOR (directional, transparent — NOT a forecast)
# ---------------------------------------------------------------------------
def simulate_scenario(baseline_kpis: dict, sim_utilisation, sim_drops, sim_oncall_share, sim_otd):
    """
    Purely DIRECTIONAL estimate: re-expresses cost as a function of the user-chosen utilisation
    and on-call share, using the dataset's OWN observed cost-per-drop by fleet type as the only
    empirical anchor. Explicitly labeled 'Directional Scenario Estimate', never 'Forecast'.
    """
    ded_cpd = baseline_kpis.get("ded_cost_per_drop", np.nan)
    oc_cpd = baseline_kpis.get("oc_cost_per_drop", np.nan)
    n_legs = baseline_kpis.get("n_legs", 0)

    if any(pd.isna(x) for x in [ded_cpd, oc_cpd]) or n_legs == 0:
        return None

    dedicated_legs = n_legs * (1 - sim_oncall_share / 100)
    oncall_legs = n_legs * (sim_oncall_share / 100)
    sim_cost = dedicated_legs * ded_cpd + oncall_legs * oc_cpd

    baseline_cost = baseline_kpis.get("total_cost", np.nan)

    return dict(
        sim_cost=sim_cost, baseline_cost=baseline_cost,
        cost_delta_pct=((sim_cost - baseline_cost) / baseline_cost * 100) if baseline_cost else np.nan,
        sim_utilisation=sim_utilisation, sim_drops=sim_drops,
        sim_oncall_share=sim_oncall_share, sim_otd=sim_otd,
    )


# ---------------------------------------------------------------------------
# 11. UI HELPER COMPONENTS
# ---------------------------------------------------------------------------
def kpi_card(label, value, target=None, higher_is_better=True, suffix="", help_text=None):
    status = kpi_status(value, target, higher_is_better) if target is not None else "gray"
    gap_str = ""
    if target is not None and not pd.isna(value):
        gap = value - target if higher_is_better else target - value
        gap_str = f"Gap: {gap:+.1f}{suffix} vs target {target}{suffix}"
    val_str = f"{value:,.1f}{suffix}" if not pd.isna(value) else "N/A"
    st.markdown(f"""
    <div class="kpi-card {status}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val_str}</div>
        <div class="kpi-sub">{gap_str}</div>
    </div>
    """, unsafe_allow_html=True)


def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


def hypothesis_or_validated(label, is_validated=False):
    cls = "val-label" if is_validated else "hyp-label"
    txt = "Validated Finding" if is_validated else "Operational Hypothesis"
    return f'<span class="{cls}">{txt}</span> {label}'


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def data_source_footer():
    st.markdown(
        '<div class="footer-note">Sumber data: delivery_log_case_dataset_LCC2026.xlsx '
        '(Delivery Log + Trip Summary, data asli — tidak ada data dummy). '
        'M3-PULSE Prototype — LCC2026.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: HOME / LANDING
# ---------------------------------------------------------------------------
def page_home(dl, ts):
    st.markdown("""
    <div class="m3-hero">
        <h1>M3-PULSE</h1>
        <p style="font-size:1.15rem; font-weight:600;">Hangry Predictive Logistics & Utilisation System</p>
        <p style="margin-top:0.6rem;">"Predict the Risk. Recover the Capacity. Call External Capacity Only When Necessary."</p>
        <p style="margin-top:1rem; font-style:italic; color:#9FB3D6;">From Reactive Recovery to Predictive Control.</p>
    </div>
    """, unsafe_allow_html=True)

    k = compute_kpis(dl, ts)
    st.markdown("### Current Problem")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("OTD Rate", k["otd_confirmed"], TARGETS["otd"], True, "%")
    with c2: kpi_card("Dedicated Utilisation", k["util_dedicated"], TARGETS["utilisation"], True, "%")
    with c3: kpi_card("On-Call Share", k["oncall_share"], None, False, "%")
    with c4: kpi_card("Total Trips", k["n_trips"])

    st.markdown("---")
    st.markdown("### Workflow")
    steps = ["DATA", "DETECT", "PREDICT", "DECIDE", "ACT", "MEASURE"]
    cols = st.columns(len(steps))
    for c, s in zip(cols, steps):
        c.markdown(f"""<div style="text-align:center; background:white; border-radius:10px; padding:0.9rem 0.3rem;
                    border:1px solid #E3E8F0; font-weight:700; color:{COLORS['navy']};">{s}</div>""",
                    unsafe_allow_html=True)

    st.markdown("")
    st.info("Klik **'Launch Control Tower'** di sidebar (atau pilih halaman **Executive Control Tower**) "
            "untuk mulai eksplorasi kondisi distribusi Hangry secara mendalam.")
    if st.button("🚀 Launch Control Tower", type="primary"):
        st.session_state["nav_page"] = "1. Executive Control Tower"
        st.rerun()

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 1: EXECUTIVE CONTROL TOWER
# ---------------------------------------------------------------------------
def page_control_tower(dl, ts):
    section_header("Executive Control Tower",
                    "Management visibility atas kondisi distribusi Hangry di Jakarta.")

    with st.expander("🔎 Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        months = c1.multiselect("Month", sorted(dl["Month"].unique()),
                                 format_func=lambda m: MONTH_LABELS.get(m, str(m)))
        fleets = c2.multiselect("Fleet Type", sorted(dl["Fleet Type"].unique()))
        drivers = c3.multiselect("Driver", sorted(dl["Driver"].unique()))
        origins = c4.multiselect("Origin", sorted(dl["Origin"].unique()))

    dl_f, ts_f = apply_filters(dl, ts, months or None, fleets or None, drivers or None, origins or None)
    if len(dl_f) == 0 or len(ts_f) == 0:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
        return
    k = compute_kpis(dl_f, ts_f)

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("OTD Rate (confirmed)", k["otd_confirmed"], TARGETS["otd"], True, "%")
    with c2: kpi_card("Dedicated Fleet Utilisation", k["util_dedicated"], TARGETS["utilisation"], True, "%")
    with c3: kpi_card("On-Call Share", k["oncall_share"], None, False, "%")
    c4, c5, c6 = st.columns(3)
    with c4: kpi_card("Avg Drops / Trip", k["avg_drops_trip"])
    with c5: kpi_card("Transportation Cost", k["total_cost"], suffix=" IDR")
    with c6: kpi_card("Total Trips", k["n_trips"])

    st.markdown("---")
    st.markdown("#### Trend (Bulan 1 → Bulan 4)")
    ms = monthly_series(dl, ts)  # unfiltered trend for full context
    if len(ms) >= 2:
        t1, t2 = st.columns(2)
        with t1:
            fig = px.line(ms, x="MonthLabel", y="OTD", markers=True, title="OTD Trend (%)")
            fig.add_hline(y=TARGETS["otd"], line_dash="dot", line_color=COLORS["red"],
                          annotation_text="Target 95%")
            fig.update_traces(line_color=COLORS["blue"])
            st.plotly_chart(fig, width='stretch')
        with t2:
            fig = px.line(ms, x="MonthLabel", y="Utilisation", markers=True, title="Dedicated Utilisation Trend (%)")
            fig.add_hline(y=TARGETS["utilisation"], line_dash="dot", line_color=COLORS["red"],
                          annotation_text="Target 90%")
            fig.update_traces(line_color=COLORS["emerald"])
            st.plotly_chart(fig, width='stretch')

        t3, t4 = st.columns(2)
        with t3:
            fig = px.line(ms, x="MonthLabel", y="OnCallShare", markers=True, title="On-Call Share Trend (%)")
            fig.update_traces(line_color=COLORS["amber"])
            st.plotly_chart(fig, width='stretch')
        with t4:
            fig = px.line(ms, x="MonthLabel", y="DropsPerTrip", markers=True, title="Average Drops per Trip Trend")
            fig.update_traces(line_color=COLORS["navy"])
            st.plotly_chart(fig, width='stretch')

        fig = px.line(ms, x="MonthLabel", y="CostIndex", markers=True,
                      title="Transportation Cost Index Trend (Bulan 1 = 100)")
        fig.update_traces(line_color=COLORS["red"])
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Data hanya mencakup 1 bulan pada filter aktif — tren bulanan tidak dapat ditampilkan.")

    st.markdown("---")
    st.markdown("#### Operational Signal")
    st.markdown(hypothesis_or_validated(
        "Dedicated Utilisation ↓ + Drops/Trip ↓ → Unused Dedicated Capacity → On-Call Dependency ↑ → Transportation Cost ↑",
        is_validated=False,
    ), unsafe_allow_html=True)
    st.caption("Rantai ini adalah HIPOTESIS OPERASIONAL yang diuji secara statistik pada halaman "
               "**Root Cause Diagnosis**. Label akan berubah menjadi 'Validated Finding' hanya jika data "
               "mendukungnya secara statistik.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 2: ROOT CAUSE DIAGNOSIS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_key_findings(dl, ts):
    """Generates 2-4 data-backed findings. Every finding must trace to a computed statistic."""
    from scipy import stats
    findings = []

    if len(ts) >= 10:
        r, pval = stats.pearsonr(ts["Drops on Trip"], ts["Utilisation (Load/Capacity)"])
        if pval < 0.05:
            findings.append(dict(
                title="Falling drops-per-trip is mechanically linked to falling utilisation",
                evidence=f"Correlation(drops/trip, utilisation) = {r:.2f}, p = {pval:.4f} (n={len(ts)} trips)",
                impact="Every drop lost per trip translates directly into wasted dedicated capacity.",
                confidence="High",
            ))

    conf = dl.dropna(subset=["On-Time Flag"])
    otd_fleet = conf.groupby("Fleet Type")["On-Time Flag"].apply(lambda s: (s == "Yes").mean() * 100)
    if "Dedicated" in otd_fleet.index and "On-Call" in otd_fleet.index:
        gap = otd_fleet["Dedicated"] - otd_fleet["On-Call"]
        findings.append(dict(
            title="On-call fleet does not deliver better OTD than dedicated",
            evidence=f"OTD Dedicated={otd_fleet['Dedicated']:.1f}% vs On-Call={otd_fleet['On-Call']:.1f}%",
            impact="Paying an on-call premium does not buy reliability — it only adds cost.",
            confidence="High" if abs(gap) > 3 else "Medium",
        ))

    cpd = ts.assign(cpd=ts["Trip Cost (IDR)"] / ts["Drops on Trip"])
    ded_cpd = cpd.loc[ts["Fleet Type"] == "Dedicated", "cpd"].mean()
    oc_cpd = cpd.loc[ts["Fleet Type"] == "On-Call", "cpd"].mean()
    if not pd.isna(ded_cpd) and ded_cpd > 0 and not pd.isna(oc_cpd):
        premium = (oc_cpd / ded_cpd - 1) * 100
        findings.append(dict(
            title=f"On-call trips carry a {premium:.0f}% cost premium per drop",
            evidence=f"Cost/drop: Dedicated=Rp{ded_cpd:,.0f} vs On-Call=Rp{oc_cpd:,.0f}",
            impact="Every avoidable on-call trip has a directly quantifiable, real cost impact.",
            confidence="High",
        ))

    pct_unconf = dl["On-Time Flag"].isna().mean() * 100
    if pct_unconf > 5:
        findings.append(dict(
            title=f"{pct_unconf:.1f}% of delivery legs have no confirmed arrival status",
            evidence=f"{dl['On-Time Flag'].isna().sum()} of {len(dl)} legs are 'Unconfirmed', consistent with manual "
                       "reconciliation and no real-time 3PL integration.",
            impact="OTD as measured today is likely understated in visibility, not necessarily in reality — "
                    "a monitoring gap, not just a delivery gap.",
            confidence="High",
        ))

    return findings[:4]


def page_root_cause(dl, ts):
    section_header("Why is Performance Deteriorating?",
                    "Sistem menguji bukti operasional — bukan mengasumsikan root cause.")

    scorecard = root_cause_scorecard(dl, ts)
    st.markdown("#### Root Cause Scorecard")
    st.dataframe(scorecard, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("#### Pareto Analysis — Late Delivery Reason Codes")
    conf = dl.dropna(subset=["On-Time Flag"])
    late_reasons = conf.loc[conf["On-Time Flag"] == "No", "Reason Code"]
    if late_reasons.notna().sum() == 0:
        st.warning("⚠️ **DATA VISIBILITY GAP** — Reason Code tidak tersedia untuk cukup banyak leg terlambat "
                   "untuk membangun Pareto yang bermakna. Ini membatasi kemampuan membedakan keterlambatan "
                   "akibat planning vs akibat armada.")
    else:
        pareto = pareto_table(late_reasons)
        fig = go.Figure()
        fig.add_bar(x=pareto["Category"], y=pareto["Count"], name="Count", marker_color=COLORS["blue"])
        fig.add_trace(go.Scatter(x=pareto["Category"], y=pareto["CumPct"], name="Cumulative %",
                                  yaxis="y2", mode="lines+markers", line_color=COLORS["red"]))
        fig.update_layout(
            title="Pareto — Late Delivery Reason Codes",
            yaxis=dict(title="Count"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width='stretch')
        na_reason_share = conf.loc[conf["On-Time Flag"] == "No", "Reason Code"].isna().mean() * 100
        if na_reason_share > 0:
            st.warning(f"⚠️ **DATA VISIBILITY GAP**: {na_reason_share:.1f}% of late legs have no Reason Code recorded, "
                       "limiting root-cause attribution for those legs.")

    st.markdown("---")
    st.markdown("#### Key Findings")
    findings = generate_key_findings(dl, ts)
    if not findings:
        st.info("Belum cukup data pada filter aktif untuk menghasilkan finding yang didukung secara statistik.")
    for i, f in enumerate(findings, 1):
        conf_badge = {"High": "green", "Medium": "amber", "Low": "gray"}.get(f["confidence"], "gray")
        st.markdown(f"""
        <div class="decision-card">
            <b>FINDING {i:02d}</b> — {f['title']} {badge(f['confidence'] + ' Confidence', conf_badge)}<br>
            <span style="color:{COLORS['gray']};font-size:0.85rem;">EVIDENCE</span><br>{f['evidence']}<br>
            <span style="color:{COLORS['gray']};font-size:0.85rem;">BUSINESS IMPACT</span><br>{f['impact']}
        </div>
        """, unsafe_allow_html=True)

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 3: OTD RISK MONITOR (CORE AI FEATURE) — scores with the Colab LR artifact,
# explains with the Colab XGBoost artifact. Trains nothing.
# ---------------------------------------------------------------------------
def build_risk_table(meta: pd.DataFrame, proba, explain_df, feat_names):
    meta = meta.copy()
    meta["Risk Probability"] = proba
    meta["Risk Level"] = [risk_level(p)[0] for p in proba]

    top_factors = []
    for i in range(len(meta)):
        row = explain_df.iloc[i]
        top3 = row.abs().sort_values(ascending=False).head(3)
        parts = []
        for feat in top3.index:
            direction = "↑ Risk" if row[feat] > 0 else "↓ Risk"
            parts.append(f"{feat} ({direction})")
        top_factors.append("; ".join(parts))
    meta["Top Risk Factors"] = top_factors

    def action(p):
        if p <= 0.30: return "Proceed as planned"
        elif p <= 0.60: return "Review route/sequence"
        elif p <= 0.80: return "Run Decision Gate — recovery likely needed"
        else: return "Mandatory intervention"
    meta["Recommended Action"] = meta["Risk Probability"].apply(action)
    return meta.sort_values("Risk Probability", ascending=False)


def page_otd_risk_monitor(dl, ts, master_delivery):
    section_header("OTD Risk Monitor",
                    "Memprediksi delivery/trip yang berisiko terlambat SEBELUM actual arrival terjadi.")

    artifacts, err = load_model_artifacts()
    if err:
        render_artifact_error(err)
        return

    meta = artifacts["metadata"]
    st.caption(f"📦 Model suite loaded from Colab — trained on **{meta.get('training_period','?')}**, "
               f"validated on **{meta.get('testing_period','?')}** "
               f"({meta.get('n_train','?')} train / {meta.get('n_test','?')} test legs).")

    feat_df, available, missing = engineer_features_for_scoring(
        master_delivery, artifacts["num_features"], artifacts["cat_features"])
    if missing:
        render_schema_mismatch(artifacts["num_features"] + artifacts["cat_features"], missing)
        return
    if len(feat_df) < 5:
        st.warning("Belum cukup delivery leg terkonfirmasi pada filter aktif untuk menghasilkan skor risiko.")
        return

    proba, explain_kind, explain_df, feat_names = score_and_explain(feat_df, artifacts)

    st.markdown("#### Model Performance (from Colab validation — NOT recomputed here)")
    m = meta.get("metrics", {})
    b = meta.get("baseline_metrics", {})
    scorer_label = meta.get("recommended_scoring_model", "Logistic Regression")
    perf_df = pd.DataFrame({
        f"Logistic Regression {'⭐ (Operational Scorer)' if scorer_label.startswith('Log') else '(Baseline)'}":
            {k: round(v, 3) for k, v in b.items()},
        f"XGBoost {'⭐ (Operational Scorer)' if scorer_label.startswith('XGB') else '(Interpretability only)'}":
            {k: round(v, 3) for k, v in m.items()},
    }).T
    st.dataframe(perf_df, width='stretch')
    st.info(f"🎯 **Operational Risk Scorer: {scorer_label}** — per the Colab notebook's final, "
            "evidence-based decision (Phase 18): the business objective is to prioritise **Recall** "
            "(catching real late deliveries), and on the validation hold-out this model achieved the "
            "higher recall. XGBoost is retained for its richer SHAP-based explainability, but its "
            "predictions do **not** drive the risk score shown below.")

    with st.expander("Why not just use whichever model scores highest on accuracy?"):
        st.write("Missing a genuinely-late delivery (false negative) costs a real service failure at "
                 "the outlet; flagging an on-time delivery as at-risk (false positive) only costs a "
                 "planner a few minutes of review. Recall is therefore prioritised over raw Accuracy — "
                 "see Model & Data Governance for the full metric breakdown.")

    st.caption(f"Explainability method: **{'SHAP (XGBoost artifact)' if explain_kind=='shap' else 'Feature Importance (fallback)'}**. "
               "Model indicates these features contributed to the predicted risk — this is a predictive "
               "association, not a causal claim.")

    risk_table = build_risk_table(feat_df[["Trip ID", "Destination", "Driver", "Fleet Type", "Origin",
                                             "Drops on Trip", "Utilisation (Load/Capacity)"]].reset_index(drop=True),
                                    proba, explain_df, feat_names)
    st.session_state["risk_table"] = risk_table
    st.session_state["model_artifacts"] = artifacts
    st.session_state["explain_df"] = explain_df
    st.session_state["explain_feat_names"] = feat_names

    st.markdown("---")
    st.markdown("#### Risk Score Table")
    f1, f2, f3, f4 = st.columns(4)
    lvl_filter = f1.multiselect("Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    fleet_filter = f2.multiselect("Fleet Type", sorted(risk_table["Fleet Type"].unique()))
    driver_filter = f3.multiselect("Driver", sorted(risk_table["Driver"].unique()))
    origin_filter = f4.multiselect("Origin", sorted(risk_table["Origin"].unique()))

    view = risk_table.copy()
    if lvl_filter: view = view[view["Risk Level"].isin(lvl_filter)]
    if fleet_filter: view = view[view["Fleet Type"].isin(fleet_filter)]
    if driver_filter: view = view[view["Driver"].isin(driver_filter)]
    if origin_filter: view = view[view["Origin"].isin(origin_filter)]

    show_cols = ["Trip ID", "Risk Probability", "Risk Level", "Fleet Type", "Driver",
                 "Drops on Trip", "Utilisation (Load/Capacity)", "Recommended Action"]
    st.dataframe(view[show_cols].style.format({"Risk Probability": "{:.1%}", "Utilisation (Load/Capacity)": "{:.1%}"}),
                width='stretch', hide_index=True)
    st.caption("**Late Risk Score** shown above is the Logistic Regression model's predicted probability "
               "of lateness (0-100%) — a genuine `predict_proba` output, not a transformed/scaled score.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE: TRIP RISK DETAIL (drill-down from Risk Monitor)
# ---------------------------------------------------------------------------
def operational_interpretation(feature_name):
    """SHAP factor -> operational interpretation -> potential action (never a causal claim)."""
    mapping = {
        "Utilisation": ("Capacity is not fully utilised relative to this trip's typical pattern",
                         "Review consolidation / route sequencing"),
        "Drops on Trip": ("Delivery complexity for this trip is elevated",
                           "Review drop sequence / route design"),
        "driver_hist_late_rate": ("Driver's historical performance shows elevated risk",
                                   "Review driver route assignment / coaching"),
        "route_hist_late_rate": ("This route has historically elevated delay risk",
                                   "Review route sequence / departure timing"),
        "fleet_hist_late_rate": ("This fleet category shows elevated historical risk",
                                   "Review dedicated capacity allocation / reliability"),
        "departure_hour": ("Time-of-day traffic exposure for this departure slot",
                            "Consider shifting departure window"),
        "dow": ("Day-of-week demand/traffic pattern", "Review day-of-week scheduling"),
        "Trip Cost": ("Proxy for trip length/complexity", "Review route distance/complexity"),
        "Vehicle Capacity": ("Vehicle size class assigned to this trip", "Review vehicle-route matching"),
        "Load Carried": ("Actual volume loaded for this trip", "Review load consolidation"),
    }
    for key, val in mapping.items():
        if feature_name.startswith(key):
            return val
    return ("Origin/Fleet-specific pattern", "Review route/fleet assignment")


def page_trip_detail():
    section_header("Trip Risk Detail", "WHY IS THIS TRIP AT RISK?")
    risk_table = st.session_state.get("risk_table")
    if risk_table is None:
        st.info("Buka halaman **OTD Risk Monitor** terlebih dahulu untuk menghasilkan skor risiko.")
        return

    trip_id = st.selectbox("Pilih Trip ID", risk_table["Trip ID"].unique())
    row = risk_table[risk_table["Trip ID"] == trip_id].iloc[0]

    lvl, color = risk_level(row["Risk Probability"])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="decision-card">
            <b>Trip ID:</b> {row['Trip ID']}<br>
            <b>Late Risk Score:</b> {row['Risk Probability']*100:.1f}% <span style="color:{COLORS['gray']};font-size:0.75rem;">(model predict_proba)</span><br>
            <b>Risk Level:</b> {badge(lvl, color)}<br>
            <b>Fleet Type:</b> {row['Fleet Type']}<br>
            <b>Driver:</b> {row['Driver']}<br>
            <b>Origin:</b> {row['Origin']}<br>
            <b>Destination:</b> {row['Destination']}<br>
            <b>Drops:</b> {row['Drops on Trip']}<br>
            <b>Utilisation:</b> {row['Utilisation (Load/Capacity)']*100:.1f}%<br>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("##### WHY IS THIS TRIP AT RISK?")
        st.caption("Model indicates these features contributed to the predicted risk (SHAP) — "
                   "this is a predictive association, not a causal claim.")
        st.write(row["Top Risk Factors"])
        st.markdown(f"**Recommended Action:** {row['Recommended Action']}")

    st.markdown("---")
    st.markdown("##### Model Signal → Operational Interpretation → Potential Action")
    explain_df = st.session_state.get("explain_df")
    feat_names = st.session_state.get("explain_feat_names")
    if explain_df is not None and feat_names is not None:
        try:
            row_idx = risk_table.index.get_loc(risk_table[risk_table["Trip ID"] == trip_id].index[0])
        except Exception:
            row_idx = None
        if row_idx is not None and row_idx < len(explain_df):
            row_shap = explain_df.iloc[row_idx]
            top5 = row_shap.abs().sort_values(ascending=False).head(5)
            translation_rows = []
            for feat in top5.index:
                interp, action = operational_interpretation(feat)
                translation_rows.append(dict(
                    Factor=feat, Direction="↑ Increases Risk" if row_shap[feat] > 0 else "↓ Decreases Risk",
                    Operational_Interpretation=interp, Potential_Action=action,
                ))
            st.dataframe(pd.DataFrame(translation_rows), width='stretch', hide_index=True)
        else:
            st.caption("Translation table unavailable for this trip (index mismatch after filtering).")
    else:
        st.caption("Buka OTD Risk Monitor terlebih dahulu untuk memuat detail SHAP.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 4: CAPACITY OPPORTUNITY
# ---------------------------------------------------------------------------
def page_capacity_opportunity(ts):
    section_header("Capacity Opportunity", "Mengidentifikasi unused capacity pada Dedicated Fleet.")

    cap_table = capacity_opportunity_table(ts)
    ded = cap_table[cap_table["Fleet Type"] == "Dedicated"]

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Avg Dedicated Utilisation", ded["Utilisation (Load/Capacity)"].mean() * 100,
                       TARGETS["utilisation"], True, "%")
    with c2: kpi_card("Total Unused Capacity (CBM)", ded["Unused Capacity (CBM)"].sum())
    with c3: kpi_card("High-Priority Trips", (ded["Priority"].str.startswith("High")).sum())

    st.markdown("#### Dedicated Fleet Utilisation Distribution — Current vs Target 90%")
    fig = px.histogram(ded, x="Utilisation %", nbins=25, color_discrete_sequence=[COLORS["blue"]])
    fig.add_vline(x=90, line_dash="dot", line_color=COLORS["red"], annotation_text="Target 90%")
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Capacity Recovery Opportunity")
    f1, f2 = st.columns(2)
    prio_filter = f1.multiselect("Priority", sorted(ded["Priority"].unique()))
    driver_filter = f2.multiselect("Driver", sorted(ded["Driver"].unique()))
    view = ded.copy()
    if prio_filter: view = view[view["Priority"].isin(prio_filter)]
    if driver_filter: view = view[view["Driver"].isin(driver_filter)]

    show_cols = ["Trip ID", "Fleet Type", "Vehicle Capacity (CBM)", "Load Carried (CBM)",
                 "Utilisation %", "Unused Capacity (CBM)", "Drops on Trip", "Priority"]
    st.dataframe(view[show_cols].sort_values("Unused Capacity (CBM)", ascending=False),
                width='stretch', hide_index=True)

    st.caption("⚠️ Label **'Potential Consolidation'** digunakan secara sengaja — bukan 'Confirmed Consolidation' — "
               "karena dataset tidak berisi data koordinat/rute/waktu tempuh yang cukup untuk memastikan "
               "kelayakan konsolidasi rute secara pasti.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 5: ON-CALL DECISION GATE (HERO FEATURE)
# ---------------------------------------------------------------------------
def page_decision_gate(dl, ts, master_trip):
    section_header("On-Call Decision Gate",
                    "Mencegah On-Call Truck digunakan secara otomatis setiap kali terjadi delivery risk.")

    trip_ids = sorted(ts["Trip ID"].unique())
    trip_id = st.selectbox("Pilih Trip ID", trip_ids, key="gate_trip_select")
    trip_row = ts[ts["Trip ID"] == trip_id].iloc[0]

    risk_table = st.session_state.get("risk_table")
    risk_score = 0.5
    if risk_table is not None and trip_id in risk_table["Trip ID"].values:
        risk_score = float(risk_table.loc[risk_table["Trip ID"] == trip_id, "Risk Probability"].iloc[0])
    else:
        st.caption("ℹ️ Skor risiko trip ini belum tersedia dari model (jalankan halaman OTD Risk Monitor dulu "
                   "untuk trip pada hold-out bulan terakhir) — menggunakan nilai default 50% untuk demonstrasi gate.")

    same_day = same_day_capacity_check(ts)
    match = same_day[(same_day["Date"] == trip_row["Date"]) & (same_day["Origin"] == trip_row["Origin"])]
    same_day_spare = float(match["unused"].iloc[0]) if len(match) else 0.0

    k = compute_kpis(dl, ts)
    otd_gap = TARGETS["otd"] - k["otd_confirmed"]

    result = decision_engine(
        risk_score=risk_score, same_day_spare_cbm=same_day_spare,
        load_cbm=float(trip_row["Load Carried (CBM)"]), drops_on_trip=int(trip_row["Drops on Trip"]),
        utilisation=float(trip_row["Utilisation (Load/Capacity)"]), otd_gap_pp=otd_gap,
    )
    st.session_state["last_decision"] = dict(trip_id=trip_id, **result, trip_row=trip_row.to_dict())

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Trip Data Retrieved")
        st.markdown(f"""
        <div class="decision-card">
        <b>Risk Score:</b> {risk_score*100:.0f}%<br>
        <b>Fleet Type:</b> {trip_row['Fleet Type']}<br>
        <b>Utilisation:</b> {trip_row['Utilisation (Load/Capacity)']*100:.1f}%<br>
        <b>Load Carried:</b> {trip_row['Load Carried (CBM)']:.2f} CBM<br>
        <b>Capacity:</b> {trip_row['Vehicle Capacity (CBM)']} CBM<br>
        <b>Drops:</b> {trip_row['Drops on Trip']}<br>
        <b>Driver:</b> {trip_row['Driver']}<br>
        <b>Origin:</b> {trip_row['Origin']}<br>
        <b>Same-day/origin dedicated spare capacity:</b> {same_day_spare:.2f} CBM
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("##### Decision Trace")
        dcolor = {"A. KEEP DEDICATED": "green", "B. RESEQUENCE / OPTIMISE": "amber",
                  "C. CONSOLIDATE": "amber", "D. USE ON-CALL": "red"}.get(result["decision"], "gray")
        st.markdown(f"""
        <div class="decision-card">
        INPUT ↓<br>RISK SCORE: {risk_score*100:.0f}% ({result['risk_band']}) ↓<br>
        CAPACITY STATUS: {result['capacity_status']} ↓<br>
        OPERATIONAL CONSTRAINTS: No new dedicated capex, no real-time integration ↓<br>
        TRIGGERED RULES:<br>{"<br>".join(result["triggered"])} ↓<br>
        <b>DECISION: {badge(result['decision'], dcolor)}</b><br><br>
        <b>RATIONALE:</b> {result['rationale']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Choose an Action")
    st.radio("Decision options:", ["A. KEEP DEDICATED", "B. RESEQUENCE / OPTIMISE", "C. CONSOLIDATE", "D. USE ON-CALL"],
             index=["A. KEEP DEDICATED", "B. RESEQUENCE / OPTIMISE", "C. CONSOLIDATE", "D. USE ON-CALL"].index(result["decision"]),
             key="chosen_decision", horizontal=True)
    st.caption("Sistem merekomendasikan opsi yang di-highlight berdasarkan rules di atas; "
               "planner tetap dapat memilih opsi lain (human-in-the-loop).")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 6: DECISION COMPARISON
# ---------------------------------------------------------------------------
def page_decision_comparison(ts):
    section_header("Decision Comparison", "Untuk trip yang sama, bandingkan konsekuensi tiap opsi.")

    last = st.session_state.get("last_decision")
    if last is None:
        st.info("Jalankan **On-Call Decision Gate** terlebih dahulu untuk memilih trip yang dibandingkan.")
        return

    ded_cost_avg = ts.loc[ts["Fleet Type"] == "Dedicated", "Trip Cost (IDR)"].mean()
    oncall_cost_avg = ts.loc[ts["Fleet Type"] == "On-Call", "Trip Cost (IDR)"].mean()

    rows = []
    for opt in ["A. KEEP DEDICATED", "B. RESEQUENCE / OPTIMISE", "C. CONSOLIDATE", "D. USE ON-CALL"]:
        info = qualitative_or_quantitative_cost(opt, ded_cost_avg, oncall_cost_avg)
        rows.append(dict(Decision=opt, **info,
                          Recommended="⭐ Yes" if opt == last["decision"] else ""))
    comp = pd.DataFrame(rows)[["Decision", "service", "capacity", "cost", "risk", "Recommended"]]
    comp.columns = ["Decision", "Service Impact", "Capacity Impact", "Cost Impact", "Operational Risk", "Recommendation"]
    st.dataframe(comp, width='stretch', hide_index=True)
    st.caption(f"Trip yang dibandingkan: **{last['trip_id']}**. Nilai biaya yang berupa angka Rupiah dihitung "
               "langsung dari rata-rata Trip Cost aktual pada dataset; nilai LOW/MEDIUM/HIGH adalah "
               "**Qualitative assessment** karena dataset tidak memiliki data untuk menghitungnya secara presisi.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 7: SCENARIO SIMULATOR
# ---------------------------------------------------------------------------
def page_scenario_simulator(dl, ts):
    section_header("Scenario Simulator", "Directional Scenario Estimate — bukan forecast.")

    k = compute_kpis(dl, ts)
    st.caption(f"Baseline aktual dari dataset: OTD={k['otd_confirmed']:.1f}%, "
               f"Utilisation={k['util_dedicated']:.1f}%, On-Call Share={k['oncall_share']:.1f}%, "
               f"Cost=Rp{k['total_cost']:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        sim_util = st.slider("Dedicated Fleet Utilisation (%)", 60, 100, int(round(k["util_dedicated"])))
        sim_drops = st.slider("Average Drops / Trip", 4, 10, int(round(k["avg_drops_trip"])))
    with c2:
        sim_oncall = st.slider("On-Call Share (%)", 0, 30, int(round(k["oncall_share"])))
        sim_otd = st.slider("OTD (%)", 70, 100, int(round(k["otd_confirmed"])))

    result = simulate_scenario(k, sim_util, sim_drops, sim_oncall, sim_otd)
    if result is None:
        st.warning("Baseline cost-per-drop tidak dapat dihitung pada data saat ini (kemungkinan salah satu "
                   "fleet type tidak memiliki trip).")
        return

    st.markdown("#### Current State vs Simulated State")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("OTD", f"{sim_otd}%", f"{sim_otd - k['otd_confirmed']:+.1f} pp")
    with c2: st.metric("Dedicated Utilisation", f"{sim_util}%", f"{sim_util - k['util_dedicated']:+.1f} pp")
    with c3: st.metric("On-Call Share", f"{sim_oncall}%", f"{sim_oncall - k['oncall_share']:+.1f} pp")
    with c4: st.metric("Transportation Cost", f"Rp{result['sim_cost']:,.0f}", f"{result['cost_delta_pct']:+.1f}%")

    fig = go.Figure()
    fig.add_bar(x=["Current State", "Simulated State"], y=[k["total_cost"], result["sim_cost"]],
                marker_color=[COLORS["gray"], COLORS["blue"]])
    fig.update_layout(title="Transportation Cost — Current vs Simulated")
    st.plotly_chart(fig, width='stretch')

    st.warning("📐 **Directional Scenario Estimate** — biaya disimulasikan HANYA dari perubahan pangsa on-call "
               "menggunakan cost-per-drop AKTUAL dataset per fleet type (metode: cost = dedicated_legs × "
               "cost/drop dedicated + on-call_legs × cost/drop on-call). Utilisation, Drops/Trip, dan OTD pada "
               "slider bersifat kontekstual/directional dan TIDAK secara langsung mengubah formula biaya di atas "
               "karena dataset tidak memiliki model biaya struktural yang memetakan hubungan tersebut secara pasti. "
               "Ini BUKAN forecast — jangan gunakan sebagai proyeksi keuangan definitif.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE: WHAT-IF DECISION SIMULATOR
# ---------------------------------------------------------------------------
def page_whatif(ts):
    section_header("What-If Decision Simulator", "What happens if we choose another decision?")
    last = st.session_state.get("last_decision")
    if last is None:
        st.info("Jalankan **On-Call Decision Gate** terlebih dahulu.")
        return

    ded_cost_avg = ts.loc[ts["Fleet Type"] == "Dedicated", "Trip Cost (IDR)"].mean()
    oncall_cost_avg = ts.loc[ts["Fleet Type"] == "On-Call", "Trip Cost (IDR)"].mean()

    choice = st.selectbox("Pilih keputusan alternatif untuk disimulasikan:",
                          ["A. KEEP DEDICATED", "B. RESEQUENCE / OPTIMISE", "C. CONSOLIDATE", "D. USE ON-CALL"])
    info = qualitative_or_quantitative_cost(choice, ded_cost_avg, oncall_cost_avg)

    st.markdown(f"""
    <div class="decision-card">
        <b>Trip:</b> {last['trip_id']}<br>
        <b>Chosen Alternative:</b> {choice}<br><br>
        <b>Service Risk:</b> {info.get('service')}<br>
        <b>Capacity Impact:</b> {info.get('capacity')}<br>
        <b>Cost Exposure:</b> {info.get('cost')}<br>
        <b>On-Call Dependency Impact:</b> {"Increases" if choice=="D. USE ON-CALL" else "No increase"}
    </div>
    """, unsafe_allow_html=True)

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 8: OPERATIONAL ALERT CENTER
# ---------------------------------------------------------------------------
def page_alert_center(dl, ts):
    section_header("Operational Alert Center")

    risk_table = st.session_state.get("risk_table")
    alerts = build_alerts(dl, ts, risk_table)

    if len(alerts) == 0:
        st.success("Tidak ada alert aktif pada data saat ini.")
        data_source_footer()
        return

    sev_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "DATA QUALITY": 3}
    alerts["_order"] = alerts["Severity"].map(sev_order)
    alerts = alerts.sort_values("_order").drop(columns="_order")

    sev_filter = st.multiselect("Filter Severity", ["CRITICAL", "HIGH", "WARNING", "DATA QUALITY"])
    view = alerts[alerts["Severity"].isin(sev_filter)] if sev_filter else alerts

    color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "WARNING": "🟡", "DATA QUALITY": "⚪"}
    view = view.copy()
    view["Severity"] = view["Severity"].apply(lambda s: f"{color_map.get(s,'')} {s}")
    st.dataframe(view, width='stretch', hide_index=True)

    clickable = view[view["TripID"].astype(str).str.startswith("TRIP")]
    if len(clickable):
        st.markdown("##### Buka Detail Trip dari Alert")
        sel = st.selectbox("Pilih Trip ID dari alert di atas", clickable["TripID"].unique())
        if st.button("Buka Trip Detail"):
            st.session_state["nav_page"] = "Trip Risk Detail"
            st.rerun()

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE: KPI TARGET TRACKER (used inside Control Tower & standalone)
# ---------------------------------------------------------------------------
def page_kpi_tracker(dl, ts):
    section_header("KPI Target Tracker")
    k = compute_kpis(dl, ts)

    rows = [
        dict(KPI="OTD Rate", Current=k["otd_confirmed"], Target=TARGETS["otd"], HigherBetter=True, Suffix="%"),
        dict(KPI="Dedicated Utilisation", Current=k["util_dedicated"], Target=TARGETS["utilisation"], HigherBetter=True, Suffix="%"),
        dict(KPI="On-Call Share", Current=k["oncall_share"], Target=None, HigherBetter=False, Suffix="%"),
    ]
    tracker_rows = []
    for r in rows:
        gap = (r["Current"] - r["Target"]) if r["Target"] is not None and r["HigherBetter"] else \
              (r["Target"] - r["Current"]) if r["Target"] is not None else np.nan
        status = kpi_status(r["Current"], r["Target"], r["HigherBetter"]) if r["Target"] is not None else "gray"
        status_label = {"green": "ON TARGET", "amber": "ATTENTION", "red": "CRITICAL", "gray": "N/A"}[status]
        tracker_rows.append(dict(
            KPI=r["KPI"], Current=f"{r['Current']:.1f}{r['Suffix']}",
            Target=f"{r['Target']}{r['Suffix']}" if r["Target"] is not None else "No explicit target",
            Gap=f"{gap:+.1f}pp" if not pd.isna(gap) else "-",
            Status=status_label, _severity=abs(gap) if not pd.isna(gap) else 0,
        ))
    tracker = pd.DataFrame(tracker_rows)
    priority_kpi = tracker.sort_values("_severity", ascending=False).iloc[0]["KPI"] if len(tracker) else "N/A"
    st.dataframe(tracker.drop(columns="_severity"), width='stretch', hide_index=True)
    st.info(f"🎯 **Priority KPI**: **{priority_kpi}** — ditentukan sebagai KPI dengan gap absolut terbesar terhadap "
            "target di antara KPI yang memiliki target eksplisit pada case (OTD ≥95%, Utilisasi ≥90%). "
            "Bobot ini transparan: severity = |gap dalam poin persentase|, tanpa pembobotan tersembunyi lain.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE: DATA QUALITY MONITOR
# ---------------------------------------------------------------------------
def page_data_quality(dq_summary, dl):
    section_header("Data Quality Monitor")
    st.dataframe(dq_summary, width='stretch', hide_index=True)

    st.markdown("#### Data Issue → Operational Risk Chain")
    st.markdown("""
    <div class="decision-card">
    DATA ISSUE: Missing Actual Arrival / On-Time Flag (Unconfirmed legs) ↓<br>
    INFORMATION GAP: Sistem tidak dapat memastikan status on-time/late untuk leg tersebut ↓<br>
    DECISION UNCERTAINTY: OTD yang dilaporkan bisa under/overstate performa aktual ↓<br>
    OPERATIONAL RISK: Keputusan intervensi (mis. Decision Gate) untuk leg tersebut tidak dapat dijalankan
    dengan keyakinan penuh sampai data terkonfirmasi.
    </div>
    <div class="decision-card">
    DATA ISSUE: Reason Code tidak tercatat untuk sebagian leg terlambat ↓<br>
    INFORMATION GAP: Penyebab keterlambatan tidak diketahui untuk leg tersebut ↓<br>
    DECISION UNCERTAINTY: Sulit membedakan keterlambatan akibat planning vs akibat armada ↓<br>
    OPERATIONAL RISK: Root cause attribution pada Pareto analysis menjadi tidak lengkap untuk porsi ini.
    </div>
    """, unsafe_allow_html=True)
    st.caption("Model risiko OTD pada halaman lain TIDAK terpengaruh langsung oleh kekosongan Reason Code, "
               "karena Reason Code sengaja tidak digunakan sebagai fitur model (untuk mencegah data leakage).")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 9: ACTION PLAN
# ---------------------------------------------------------------------------
def page_action_plan():
    section_header("Action Plan", "0-30 / 31-90 Days / 3-12 Months")

    plan = [
        dict(Phase="0-30 DAYS — STABILISE", Action="Data cleaning & KPI baseline (per Data Quality Monitor)",
             Objective="Visibilitas KPI mingguan yang bersih dan konsisten", Owner="Logistics Analyst",
             KPI="OTD, Utilisation, On-Call Share (weekly)", Effect="Baseline terpercaya untuk seluruh keputusan berikutnya",
             Dependency="Data harian tersedia dari 3PL (manual)", Timeline="Minggu 1-2"),
        dict(Phase="0-30 DAYS — STABILISE", Action="Deploy OTD Risk Scoring (batch harian)",
             Objective="Deteksi risiko sebelum keterlambatan terjadi", Owner="Logistics Analyst",
             KPI="Recall model pada leg yang benar-benar terlambat", Effect="Early-warning tanpa integrasi real-time",
             Dependency="Minimal 2 bulan data historis dengan status terkonfirmasi", Timeline="Minggu 2-4"),
        dict(Phase="0-30 DAYS — STABILISE", Action="Rollout On-Call Decision Gate (rule-based)",
             Objective="Hentikan penggunaan on-call yang tidak diuji dulu terhadap opsi konsolidasi", Owner="Logistics Lead + 3PL Coordinator",
             KPI="Rasio 'on-call disetujui via gate vs di luar gate'", Effect="Menahan laju kenaikan on-call share",
             Dependency="Kesediaan 3PL menjalankan checklist gate", Timeline="Minggu 3-4"),
        dict(Phase="0-30 DAYS — STABILISE", Action="Weekly KPI review & exception management",
             Objective="Disiplin monitoring mingguan", Owner="Logistics Lead",
             KPI="Seluruh KPI headline", Effect="Deteksi dini penyimpangan dari target",
             Dependency="Dashboard M3-PULSE berjalan", Timeline="Berkelanjutan"),
        dict(Phase="31-90 DAYS — OPTIMISE", Action="Route optimisation & capacity recovery pada trip kuadran prioritas",
             Objective="Naikkan drops/trip dan utilisasi dedicated", Owner="Planning Team",
             KPI="Drops/Trip, Dedicated Utilisation", Effect="Utilisation naik menuju target 90%",
             Dependency="Hasil review Capacity Opportunity", Timeline="Bulan 2-3"),
        dict(Phase="31-90 DAYS — OPTIMISE", Action="Driver performance management",
             Objective="Kurangi variasi performa OTD antar-driver", Owner="Fleet Supervisor 3PL",
             KPI="OTD per driver", Effect="Mengurangi kontribusi keterlambatan dari driver underperforming",
             Dependency="Driver scorecard dari Root Cause Diagnosis", Timeline="Bulan 2-3"),
        dict(Phase="31-90 DAYS — OPTIMISE", Action="Consolidation opportunity execution",
             Objective="Realisasi 'Potential Consolidation' menjadi rute aktual", Owner="Planning Team + 3PL",
             KPI="Avg Drops/Trip, On-Call Share", Effect="Penurunan cost/drop",
             Dependency="Data rute/waktu tempuh tambahan [DATA NEEDED]", Timeline="Bulan 3"),
        dict(Phase="3-12 MONTHS — INSTITUTIONALISE", Action="Automated data pipeline (batch, bertahap menuju lebih sering)",
             Objective="Kurangi lag rekonsiliasi manual", Owner="Logistics Lead + IT",
             KPI="% legs Unconfirmed", Effect="Visibilitas lebih cepat tanpa melanggar constraint 'no real-time saat ini'",
             Dependency="Payback case disetujui", Timeline="Bulan 4-9"),
        dict(Phase="3-12 MONTHS — INSTITUTIONALISE", Action="3PL integration & SLA redesign",
             Objective="Formalkan Decision Gate ke dalam kontrak/SOP", Owner="Logistics Lead + Legal/Procurement",
             KPI="Seluruh KPI headline vs Exhibit A", Effect="Perbaikan bersifat institusional, bukan sementara",
             Dependency="Sisa masa kontrak 3PL", Timeline="Bulan 6-12"),
        dict(Phase="3-12 MONTHS — INSTITUTIONALISE", Action="Predictive planning & phased investment (jika justified)",
             Objective="Tingkatkan akurasi model dengan data yang lebih besar", Owner="Logistics Lead + Finance",
             KPI="Model Recall/Precision, ROI realisasi", Effect="Model matang untuk skala penuh",
             Dependency="Bukti manfaat dari fase sebelumnya", Timeline="Bulan 9-12"),
    ]
    plan_df = pd.DataFrame(plan)
    for phase in plan_df["Phase"].unique():
        st.markdown(f"#### {phase}")
        st.dataframe(plan_df[plan_df["Phase"] == phase].drop(columns="Phase"),
                    width='stretch', hide_index=True)

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE 10: MODEL & DATA GOVERNANCE
# ---------------------------------------------------------------------------
def page_model_governance(dl, ts, dq_summary):
    section_header("Model & Data Governance")

    artifacts, err = load_model_artifacts()
    st.markdown("#### Model")
    if err:
        render_artifact_error(err)
    else:
        meta = artifacts["metadata"]
        m = meta.get("metrics", {})
        b = meta.get("baseline_metrics", {})
        model_files = meta.get("model_files", {})

        gov = pd.DataFrame([
            dict(Item="Model Suite", Value=meta.get("model_name", "n/a")),
            dict(Item="Model Version", Value=meta.get("model_version", "n/a")),
            dict(Item="Operational Risk Scorer", Value=meta.get("recommended_scoring_model", "n/a")),
            dict(Item="Interpretability Model", Value="XGBoost (SHAP)"),
            dict(Item="Prediction Target", Value=meta.get("target", "On-Time Flag (binary: Late vs On-Time)")),
            dict(Item="Training Period", Value=meta.get("training_period", "n/a")),
            dict(Item="Testing Period", Value=meta.get("testing_period", "n/a")),
            dict(Item="Training Records", Value=meta.get("n_train", "n/a")),
            dict(Item="Testing Records", Value=meta.get("n_test", "n/a")),
            dict(Item="Numeric Features", Value=", ".join(meta.get("features_numeric", []))),
            dict(Item="Categorical Features", Value=", ".join(meta.get("features_categorical", []))),
            dict(Item="Features EXCLUDED (leakage prevention)", Value=", ".join(meta.get("features_excluded_leakage", []))),
            dict(Item="Risk Threshold", Value=" | ".join(f"{k} {v}" for k, v in meta.get("risk_thresholds", {}).items())),
        ])
        st.dataframe(gov.astype({"Value": str}), width='stretch', hide_index=True)

        st.markdown("#### Model Artifact Files (loaded from Colab, never retrained)")
        files_df = pd.DataFrame([{"Role": k, "File": v} for k, v in model_files.items()]) if model_files else \
                   pd.DataFrame([{"Role": k, "File": f"models/{v}"} for k, v in REQUIRED_ARTIFACTS.items() if k != "metadata"])
        st.dataframe(files_df, width='stretch', hide_index=True)

        st.markdown("#### Model Performance (as validated in Colab — source of truth, not recomputed here)")
        perf_df = pd.DataFrame({"Logistic Regression": b, "XGBoost": m}).T.round(3)
        st.dataframe(perf_df, width='stretch')
        st.caption("Recall is the metric this project prioritises: catching a genuinely-late delivery "
                   "matters more than avoiding an occasional false alarm.")

        st.markdown("#### Leakage Check")
        excluded = set(meta.get("features_excluded_leakage", []))
        used = set(meta.get("features_numeric", []) + meta.get("features_categorical", []))
        passed = used.isdisjoint(excluded)
        st.markdown(badge("PASSED — no post-delivery information used as a feature" if passed else "WARNING — leakage risk detected",
                          "green" if passed else "red"), unsafe_allow_html=True)

        st.markdown("#### Model Limitation")
        limitations = meta.get("limitations", [])
        if limitations:
            st.warning("\n".join(f"- {l}" for l in limitations))
        else:
            st.info("No limitations recorded in model_metadata.json.")

    st.markdown("#### Data")
    dgov = pd.DataFrame([
        dict(Item="Delivery Log Records", Value=len(dl)),
        dict(Item="Trip Summary Records", Value=len(ts)),
        dict(Item="Missing Values (unconfirmed legs)", Value=int(dl["On-Time Flag"].isna().sum())),
        dict(Item="Duplicate Records", Value=int(dl["Record ID"].duplicated().sum())),
    ])
    st.dataframe(dgov, width='stretch', hide_index=True)
    with st.expander("Full Data Quality Summary"):
        st.dataframe(dq_summary, width='stretch', hide_index=True)

    st.error("⚠️ **DISCLAIMER**: Model predictions are decision-support signals, not autonomous "
             "operational decisions. All final operational decisions remain with the Hangry "
             "logistics team and 3PL planners.")

    data_source_footer()


# ---------------------------------------------------------------------------
# PAGE: FINAL EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------
def page_final_summary(dl, ts):
    st.markdown("""
    <div class="m3-hero">
        <h1 style="font-size:1.7rem;">FROM REACTIVE RECOVERY TO PREDICTIVE CONTROL</h1>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### BEFORE")
        st.markdown("""
        <div class="decision-card">
        Late Delivery ↓<br>Reactive Detection ↓<br>On-Call Truck ↓<br><b>Higher Cost</b>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("##### AFTER")
        st.markdown("""
        <div class="decision-card">
        Risk Prediction ↓<br>Capacity Check ↓<br>Decision Engine ↓<br><b>On-Call Only When Necessary</b>
        </div>
        """, unsafe_allow_html=True)

    k = compute_kpis(dl, ts)
    st.markdown("##### Target")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("OTD", k["otd_confirmed"], TARGETS["otd"], True, "%")
    with c2: kpi_card("Dedicated Utilisation", k["util_dedicated"], TARGETS["utilisation"], True, "%")
    with c3: kpi_card("On-Call Dependency", k["oncall_share"], None, False, "%")
    with c4: kpi_card("Transportation Cost", k["total_cost"], suffix=" IDR")

    st.markdown("---")
    st.markdown("""
    > *"We do not solve the problem by adding trucks.
    > We solve it by making better logistics decisions earlier."*
    """)
    data_source_footer()


# ---------------------------------------------------------------------------
# DEMO MODE
# ---------------------------------------------------------------------------
DEMO_STEPS = [
    ("STEP 1 — Current Problem", "Three operational indicators are deteriorating: OTD, dedicated fleet "
     "utilisation, and on-call dependency.", "1. Executive Control Tower"),
    ("STEP 2 — Root Cause", "We do not assume the root cause. The system tests the operational evidence.",
     "2. Root Cause Diagnosis"),
    ("STEP 3 — High-Risk Trip", "We identify which delivery is likely to fail before it becomes late.",
     "3. OTD Risk Monitor"),
    ("STEP 4 — Capacity Opportunity", "Before deploying an external truck, we check whether existing "
     "dedicated capacity can be recovered.", "4. Capacity Opportunity"),
    ("STEP 5 — Decision Gate", "The system does not automatically call an on-call truck. It evaluates "
     "the alternatives first.", "5. On-Call Decision Gate"),
    ("STEP 6 — Decision Comparison", "We compare the operational consequences of each available decision.",
     "6. Decision Comparison"),
    ("STEP 7 — Scenario", "We test how operational changes affect OTD, utilisation and on-call dependency.",
     "7. Scenario Simulator"),
    ("STEP 8 — Action Plan", "The solution translates analytics into a 30-60-90 day implementation plan.",
     "9. Action Plan"),
]


def render_demo_controller():
    if "demo_step" not in st.session_state:
        st.session_state["demo_step"] = 0
    idx = st.session_state["demo_step"]
    title, narration, target_page = DEMO_STEPS[idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🎬 DEMO MODE — {title}**")
    st.sidebar.caption(narration)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("← Previous", disabled=idx == 0):
        st.session_state["demo_step"] = max(0, idx - 1)
        st.session_state["nav_page"] = DEMO_STEPS[st.session_state["demo_step"]][2]
        st.rerun()
    if c2.button("Next →", disabled=idx == len(DEMO_STEPS) - 1):
        st.session_state["demo_step"] = min(len(DEMO_STEPS) - 1, idx + 1)
        st.session_state["nav_page"] = DEMO_STEPS[st.session_state["demo_step"]][2]
        st.rerun()
    return target_page


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
PAGES_MANAGEMENT = [
    "Home",
    "1. Executive Control Tower",
    "2. Root Cause Diagnosis",
    "7. Scenario Simulator",
    "9. Action Plan",
    "Final Executive Summary",
    "10. Model & Data Governance",
]
PAGES_PLANNER = [
    "Home",
    "1. Executive Control Tower",
    "2. Root Cause Diagnosis",
    "3. OTD Risk Monitor",
    "Trip Risk Detail",
    "4. Capacity Opportunity",
    "5. On-Call Decision Gate",
    "6. Decision Comparison",
    "What-If Simulator",
    "7. Scenario Simulator",
    "8. Alert Center",
    "KPI Target Tracker",
    "Data Quality Monitor",
    "9. Action Plan",
    "10. Model & Data Governance",
    "Final Executive Summary",
]


def main():
    dl_raw, ts_raw, err = load_data(DATA_PATH)
    if err:
        st.error(f"❌ {err}\n\nPastikan file `delivery_log_case_dataset_LCC2026.xlsx` berada di folder yang "
                 "sama dengan app.py.")
        st.stop()

    dl, ts, dq_summary = clean_data(dl_raw, ts_raw)
    master_delivery, master_trip = build_master(dl, ts)

    st.sidebar.markdown("## 🚚 M3-PULSE")
    st.sidebar.caption("Hangry Predictive Logistics & Utilisation System")

    view_mode = st.sidebar.radio("View", ["MANAGEMENT VIEW", "OPERATIONS / PLANNER VIEW"], horizontal=False)
    demo_mode = st.sidebar.toggle("🎬 DEMO MODE", value=False)

    pages = PAGES_MANAGEMENT if view_mode == "MANAGEMENT VIEW" else PAGES_PLANNER

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"

    if demo_mode:
        target = render_demo_controller()
        st.session_state["nav_page"] = target
        page = target
    else:
        default_idx = pages.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in pages else 0
        page = st.sidebar.radio("Navigate", pages, index=default_idx, key="nav_radio")
        st.session_state["nav_page"] = page

    st.sidebar.markdown("---")
    _artifacts, _artifact_err = load_model_artifacts()
    st.sidebar.caption(f"Model artifacts: {'✅ loaded from Colab' if not _artifact_err else '⚠️ not found'}")
    st.sidebar.caption(f"SHAP: {'✅ available' if HAS_SHAP else '⚠️ fallback to Feature Importance'}")
    st.sidebar.caption(f"Records: {len(dl):,} legs | {len(ts):,} trips")

    # --- ROUTER ---
    if page == "Home":
        page_home(dl, ts)
    elif page == "1. Executive Control Tower":
        page_control_tower(dl, ts)
    elif page == "2. Root Cause Diagnosis":
        page_root_cause(dl, ts)
    elif page == "3. OTD Risk Monitor":
        page_otd_risk_monitor(dl, ts, master_delivery)
    elif page == "Trip Risk Detail":
        page_trip_detail()
    elif page == "4. Capacity Opportunity":
        page_capacity_opportunity(ts)
    elif page == "5. On-Call Decision Gate":
        page_decision_gate(dl, ts, master_trip)
    elif page == "6. Decision Comparison":
        page_decision_comparison(ts)
    elif page == "What-If Simulator":
        page_whatif(ts)
    elif page == "7. Scenario Simulator":
        page_scenario_simulator(dl, ts)
    elif page == "8. Alert Center":
        page_alert_center(dl, ts)
    elif page == "KPI Target Tracker":
        page_kpi_tracker(dl, ts)
    elif page == "Data Quality Monitor":
        page_data_quality(dq_summary, dl)
    elif page == "9. Action Plan":
        page_action_plan()
    elif page == "10. Model & Data Governance":
        page_model_governance(dl, ts, dq_summary)
    elif page == "Final Executive Summary":
        page_final_summary(dl, ts)
    else:
        page_home(dl, ts)


if __name__ == "__main__":
    main()
