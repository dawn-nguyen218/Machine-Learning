"""
Sustainable Concrete Mix Design Tool
====================================
Predicts compressive strength and embodied CO2 for a concrete mix, flags
inputs outside the training range, and suggests comparable-strength mixes
with a smaller carbon footprint.

This app reproduces the deployed model from the report exactly: a Random
Forest (200 trees, random_state=42) trained on the 8 raw mix variables plus
6 engineered features, fit on the 70% training partition of the cleaned
UCI Concrete Compressive Strength dataset (1,005 unique mixes after duplicate
removal). Lower-carbon alternatives are drawn from the full dataset.

DESIGN-EXPLORATION / SCREENING AID ONLY. Predictions are estimates and do
NOT replace laboratory testing, building-code compliance, or professional
engineering judgment.

Run locally:   streamlit run app.py
Dependencies:  see requirements.txt
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- canonical schema (positional; matches both ucimlrepo id=165 and the CSV) ---
COLS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
        "coarse_agg", "fine_agg", "age", "strength"]
MATERIALS = COLS[:7]  # the seven ingredients that carry a CO2 factor

# Embodied-CO2 factors (kg CO2 / kg). Order matches MATERIALS.
# Slag and fly-ash factors are allocation-sensitive (see report factor table).
CO2_FACTORS = np.array([0.820, 0.143, 0.027, 0.0003, 0.720, 0.0046, 0.0014])

ENGINEERED = ["binder", "w_c_ratio", "w_b_ratio",
              "scm_fraction", "agg_binder_ratio", "log_age"]
FEATURES = MATERIALS + ["age"] + ENGINEERED  # 8 raw + 6 engineered = 14

DEFAULTS = dict(cement=250, slag=150, fly_ash=80, water=170,
                superplasticizer=8, coarse_agg=1000, fine_agg=750,
                age=28, target=40)

LABELS = [("cement", "Cement"), ("slag", "Blast furnace slag"),
          ("fly_ash", "Fly ash"), ("water", "Water"),
          ("superplasticizer", "Superplasticizer"),
          ("coarse_agg", "Coarse aggregate"), ("fine_agg", "Fine aggregate")]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the six physically motivated engineered features. Deterministic
    row-wise functions of the inputs, so safe to compute at predict time."""
    f = frame.copy()
    f["binder"] = f["cement"] + f["slag"] + f["fly_ash"]
    f["w_c_ratio"] = f["water"] / f["cement"]
    f["w_b_ratio"] = f["water"] / f["binder"]
    f["scm_fraction"] = (f["slag"] + f["fly_ash"]) / f["binder"]
    f["agg_binder_ratio"] = (f["coarse_agg"] + f["fine_agg"]) / f["binder"]
    f["log_age"] = np.log(f["age"])
    return f


@st.cache_resource(show_spinner="Loading data and training model...")
def load_and_train():
    """Load the UCI dataset, clean it, compute CO2 + features, and train the
    report's Random Forest on the 703-row training split."""
    df = None
    try:  # primary: pull straight from the UCI repository
        from ucimlrepo import fetch_ucirepo
        raw = fetch_ucirepo(id=165)
        df = pd.concat([raw.data.features, raw.data.targets], axis=1)
    except Exception:  # offline fallback: bundled CSV (same column order)
        df = pd.read_csv("concrete.csv")

    df = df.iloc[:, :9]
    df.columns = COLS
    df = df.drop_duplicates().reset_index(drop=True)        # 1,030 -> 1,005
    df["co2"] = df[MATERIALS].values @ CO2_FACTORS          # embodied CO2 / m3
    df = add_features(df)

    X, y = df[FEATURES], df["strength"]
    X_tr, _X_tmp, y_tr, _y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(scaler.transform(X_tr), y_tr)
    return df, scaler, model, X_tr.min(), X_tr.max()


def predict_strength(model, scaler, mix: dict, age) -> float:
    row = add_features(pd.DataFrame([{**mix, "age": age}]))[FEATURES]
    return float(model.predict(scaler.transform(row))[0])


def main():
    st.set_page_config(page_title="Sustainable Concrete Mix Design",
                       page_icon="🧱", layout="wide")
    df, scaler, model, tr_min, tr_max = load_and_train()

    st.title("Sustainable Concrete Mix Design Tool")
    st.caption("Predict compressive strength and embodied CO\u2082 for a mix, "
               "then find comparable mixes with a smaller carbon footprint.")
    st.warning(
        "**Design-exploration and screening aid only.** Predictions are "
        "estimates from a model trained on a single public dataset "
        "(UCI / Yeh 1998) and do **not** replace laboratory testing, "
        "building-code compliance, or professional engineering judgment. "
        "Validate any mix experimentally before use in construction.")

    # ---------------- sidebar inputs ----------------
    with st.sidebar:
        st.header("Mix composition (kg/m\u00b3)")
        mix = {}
        for key, label in LABELS:
            mix[key] = st.number_input(
                f"{label}  \u00b7  train range {tr_min[key]:.0f}\u2013{tr_max[key]:.0f}",
                min_value=0.0, value=float(DEFAULTS[key]), step=5.0)
        st.divider()
        target = st.number_input("Target strength (MPa)", min_value=0.0,
                                 value=float(DEFAULTS["target"]), step=1.0)
        age = st.radio("Curing age (days)", [7, 14, 28, 90], index=2,
                       horizontal=True)
        run = st.button("Run prediction", type="primary",
                        width="stretch")

    if not run:
        st.info("Set a mix in the sidebar and click **Run prediction**.")
        return

    # ---------------- core prediction ----------------
    pred = predict_strength(model, scaler, mix, age)
    co2 = float(np.array([mix[m] for m in MATERIALS]) @ CO2_FACTORS)
    eff = pred / co2 if co2 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Predicted strength @ {age} d", f"{pred:.1f} MPa")
    c2.metric("Embodied CO\u2082", f"{co2:.0f} kg/m\u00b3")
    c3.metric("Carbon efficiency", f"{eff:.3f}", help="MPa per kg CO\u2082/m\u00b3")

    if pred >= target:
        st.success(f"**Meets** the {target:.0f} MPa target "
                   f"(margin +{pred - target:.1f} MPa).")
    else:
        st.error(f"**Does not meet** the {target:.0f} MPa target "
                 f"(short by {target - pred:.1f} MPa).")

    # out-of-range / extrapolation warning
    row = add_features(pd.DataFrame([{**mix, "age": age}]))[FEATURES].iloc[0]
    oor = [c for c in FEATURES if not (tr_min[c] <= row[c] <= tr_max[c])]
    if oor:
        st.warning("Inputs outside the training range: **"
                   + ", ".join(oor) + "**. This prediction is an "
                   "extrapolation \u2014 treat it with extra caution.")

    # how carbon-intensive vs comparable-strength mixes
    similar = df[df["strength"].between(pred - 5, pred + 5)]
    if len(similar):
        pct = (similar["co2"] < co2).mean() * 100
        st.write(f"This mix is more carbon-intensive than **{pct:.0f}%** of "
                 f"dataset mixes within \u00b15 MPa of its predicted strength.")

    # ---------------- strength development ----------------
    st.subheader("Predicted strength development")
    ages = [7, 14, 28, 90]
    curve = [predict_strength(model, scaler, mix, a) for a in ages]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(ages, curve, "o-", color="#2b6cb0", linewidth=2)
    ax.axhline(target, ls="--", lw=1, color="gray",
               label=f"target {target:.0f} MPa")
    for a, s in zip(ages, curve):
        ax.annotate(f"{s:.1f}", (a, s), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("Curing age (days)")
    ax.set_ylabel("Predicted strength (MPa)")
    ax.set_xticks(ages)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Strength at ages not represented in the training data is "
               "model interpolation/extrapolation and carries higher "
               "uncertainty.")

    # ---------------- lower-carbon alternatives ----------------
    st.subheader("Lower-carbon alternatives of comparable strength")
    pool = (df[df["strength"].between(pred - 5, pred + 5) & (df["co2"] < co2)]
            .sort_values("co2").head(5))
    if len(pool):
        show = pool[["cement", "slag", "fly_ash", "water", "age",
                     "strength", "co2"]].copy()
        show.columns = ["Cement", "Slag", "Fly ash", "Water", "Age",
                        "Strength (MPa)", "CO\u2082 (kg/m\u00b3)"]
        show["Strength (MPa)"] = show["Strength (MPa)"].round(1)
        show["CO\u2082 (kg/m\u00b3)"] = show["CO\u2082 (kg/m\u00b3)"].round(0)
        st.dataframe(show.reset_index(drop=True), width="stretch")
        best = pool.iloc[0]
        st.write(
            f"Lowest-carbon comparable mix: **{best['strength']:.1f} MPa** at "
            f"**{best['co2']:.0f} kg CO\u2082/m\u00b3** (vs your {co2:.0f}) "
            f"\u2014 a **{100 * (co2 - best['co2']) / co2:.0f}% reduction** in "
            f"embodied carbon, reached with ~{best['scm_fraction'] * 100:.0f}% "
            f"supplementary cementitious material in the binder.")
    else:
        st.write("No comparable-strength mix in the dataset has lower embodied "
                 "CO\u2082 than this one.")

    st.divider()
    st.caption("Embodied-CO\u2082 factors are published LCA averages; slag and "
               "fly-ash values depend on allocation assumptions and should be "
               "read as comparative estimates, not project-specific values. "
               "Model: Random Forest (200 trees) on 14 features; reproduces the "
               "report's deployed model (test RMSE \u2248 5.26 MPa).")


if __name__ == "__main__":
    main()
