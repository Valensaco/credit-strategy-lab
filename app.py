"""Credit Strategy Lab: turn a default model into an approval decision.

Data: American Express Default Prediction (Kaggle). The repo ships only the trained
model and summary tables in app_data/; no customer-level rows.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

DATA = Path(__file__).parent / "app_data"
NAVY, BLUE, ORANGE, RED, GREEN, GREY = "#14213D", "#1F5FBF", "#C8571B", "#B3261E", "#2B8A3E", "#8A94A6"
SAMPLES = ["Train", "Test 1", "Test 2"]

st.set_page_config(page_title="Credit Strategy Lab", page_icon="📊", layout="wide")


# ---------- data ----------
@st.cache_data
def load_tables():
    names = ["threshold_grid", "auc", "rank_order", "features", "shap_importance",
             "grid_xgb", "grid_nn", "model_comparison"]
    t = {n: pd.read_csv(DATA / f"{n}.csv") for n in names}
    t["meta"] = json.loads((DATA / "meta.json").read_text())
    return t


@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model(str(DATA / "model.json"))
    return booster


T = load_tables()
META = T["meta"]
FEATURES = META["selected_features"]
CONSERVATIVE, AGGRESSIVE = META["conservative_cutoff"], META["aggressive_cutoff"]
booster = load_model()
CATEGORY = {"D": "Delinquency", "B": "Balance", "R": "Risk", "S": "Spend", "P": "Payment"}


def describe(name: str) -> str:
    """Plain-English label for an engineered feature name."""
    rules = [
        (r"^(.*)_Response_Rate_(\d+)$", "% of last {1} months at {0}"),
        (r"^(.*)_Ever_Response_(\d+)$", "ever at {0} in last {1} months"),
        (r"^(.*)_Ave_(\d+)$", "average {0}, last {1} months"),
        (r"^(.*)_Min_(\d+)$", "lowest {0}, last {1} months"),
        (r"^(.*)_Max_(\d+)$", "highest {0}, last {1} months"),
        (r"^(.*)_Sum_(\d+)$", "sum of {0}, last {1} months"),
        (r"^(.*)_Chg_(\d+)$", "% change in {0} vs {1} months ago"),
        (r"^(.*)_Apr_2018$", "latest {0}"),
    ]
    for pat, fmt in rules:
        m = re.match(pat, name)
        if m:
            base = m.group(1)
            parts = base.split("_")
            if len(parts) == 3:  # one-hot level, e.g. D_116_0
                base = f"{parts[0]}_{parts[1]} = {parts[2]}"
            return f"{CATEGORY.get(base[0], '')} · " + fmt.format(base, *m.groups()[1:])
    return name


def pct(x):
    return "–" if pd.isna(x) else f"{x:.2%}"


def num(x):
    return f"{x:,.2f}"


# ---------- sidebar ----------
st.sidebar.header("Assumptions")
limit = st.sidebar.slider("Max default rate the company accepts", 2.0, 20.0, 10.0, 0.5, format="%.1f%%",
                          help="A cut-off is 'safe' only if it stays at or under this on Train, Test 1 and Test 2.") / 100
view = st.sidebar.selectbox("Show results for", SAMPLES, index=2,
                            help="Test 2 is the most honest view: customers the model never trained on.")
st.sidebar.divider()
st.sidebar.markdown("**Revenue rule** (from the project brief)  \n"
                    "12 months × (2% of balance + 0.1% of spend) for approved customers who don't default; "
                    "defaulters earn 0.")
st.sidebar.caption("Amex anonymized and rescaled every value, so revenue is in the data's scaled units: "
                   "use it to compare strategies, not as dollars.")

G = T["threshold_grid"].copy()
G["threshold"] = G["threshold"].round(2)
G["approval_rate"] = G["n_accepted"] / G["n_total"]
worst = G.pivot(index="threshold", columns="sample", values="default_rate").max(axis=1)
safe = worst[worst <= limit].index
cur = G[G["sample"] == view].set_index("threshold")
best_t = float(safe.max()) if len(safe) else None

# ---------- header ----------
st.title("Credit Strategy Lab")
st.markdown("**Who should get a credit card?** An XGBoost model scores each customer's risk of default from "
            "13 months of card statements. This app turns that score into a decision: choose a cut-off and see "
            "how many people you approve, how many default, and how much revenue the portfolio earns.")
auc = T["auc"].set_index("sample")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers modeled", f"{META['n_customers']:,}")
c2.metric("Baseline default rate", pct(META["overall_default_rate"]))
c3.metric("Model AUC on new customers", f"{auc.loc['Test 2', 'auc']:.3f}", help="Test 2 sample")
c4.metric("Features in the model", len(FEATURES))

tab_sim, tab_score, tab_model, tab_about = st.tabs(
    ["Strategy simulator", "Score an applicant", "Model", "About"])

# ---------- tab 1: strategy simulator ----------
with tab_sim:
    t = st.slider("Approve applicants with a risk score below", 0.05, 1.0, AGGRESSIVE, 0.01, format="%.2f",
                  key="cutoff")
    t = round(t, 2)
    row, ok = cur.loc[t], worst.loc[t] <= limit

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approval rate", pct(row["approval_rate"]), f"{int(row['n_accepted']):,} approved",
              delta_color="off", delta_arrow="off")
    k2.metric("Default rate of approved", pct(row["default_rate"]), "within limit" if ok else "over limit",
              delta_color="green" if ok else "red", delta_arrow="off",
              help="Within limit = at or under the limit on Train, Test 1 and Test 2.")
    k3.metric("12-month revenue", num(row["revenue"]))
    rev_cons = cur.loc[CONSERVATIVE, "revenue"]
    k4.metric("vs. conservative strategy", f"{row['revenue'] / rev_cons:.2f}×" if rev_cons else "–",
              help=f"Revenue relative to the conservative cut-off ({CONSERVATIVE:.2f}).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cur.index, y=cur["default_rate"], name="Default rate", line=dict(color=RED, width=3)))
    fig.add_trace(go.Scatter(x=cur.index, y=cur["revenue"], name="Revenue", yaxis="y2", line=dict(color=BLUE, width=3)))
    fig.add_hline(y=limit, line=dict(color=RED, dash="dash", width=1),
                  annotation_text=f"{limit:.1%} limit", annotation_position="top left")
    if best_t is not None:
        fig.add_vrect(x0=0.05, x1=best_t, fillcolor=GREEN, opacity=0.06, line_width=0,
                      annotation_text="safe zone", annotation_position="top left")
    fig.add_vline(x=t, line=dict(color=NAVY, dash="dot"), annotation_text="your cut-off")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", plot_bgcolor="white",
                      xaxis=dict(title="Risk score cut-off (approve if below)", range=[0.05, 1.0]),
                      yaxis=dict(title="Default rate of approved", tickformat=".0%"),
                      yaxis2=dict(title="Revenue (scaled units)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    if best_t is None:
        st.warning("No cut-off keeps the default rate under this limit on all three samples.")
    else:
        b = cur.loc[best_t]
        st.info(f"**Highest safe cut-off: {best_t:.2f}.** It approves {pct(b['approval_rate'])} of applicants with a "
                f"{pct(b['default_rate'])} default rate on {view}, and the worst sample stays at {pct(worst.loc[best_t])}. "
                "Above it, revenue keeps rising but the default rate breaks the limit.")

    st.subheader("Compare strategies")
    table = []
    for name, th in [("Conservative", CONSERVATIVE), ("Aggressive", AGGRESSIVE), ("Your cut-off", t)]:
        r = cur.loc[th]
        table.append({"Strategy": name, "Cut-off": f"{th:.2f}", "Approved": f"{int(r['n_accepted']):,}",
                      "Approval rate": pct(r["approval_rate"]), f"Default rate ({view})": pct(r["default_rate"]),
                      "Worst default rate (all samples)": pct(worst.loc[th]), "Revenue": num(r["revenue"])})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("Conservative = highest cut-off with default ≤ 5% on every sample; aggressive = highest with ≤ 10%. "
               f"Revenue uses {META['balance_feature']} (balance) and {META['spend_feature']} (spend), "
               "averaged over the last 6 months.")

# ---------- tab 2: score an applicant ----------
with tab_score:
    st.markdown("Start from a **typical customer** and move the factors that matter most. "
                "The real model rescores instantly and explains the result.")
    feats = T["features"].set_index("feature")
    top = T["shap_importance"]["feature"].head(8).tolist()
    x = feats["typical"].reindex(FEATURES).to_dict()
    left, right = st.columns([1, 1.3])
    with left:
        if st.button("Reset to typical customer"):
            for f in top:
                st.session_state.pop(f"w_{f}", None)
        for f in top:
            lo, hi = float(feats.loc[f, "p01"]), float(feats.loc[f, "p99"])
            if hi <= lo:
                continue
            x[f] = st.slider(describe(f), lo, hi, float(np.clip(x[f], lo, hi)), (hi - lo) / 100,
                             format="%.3f", key=f"w_{f}", help=f"Model feature {f}; range = 1st to 99th percentile")
    dm = xgb.DMatrix(pd.DataFrame([x])[FEATURES])
    score = float(booster.predict(dm)[0])
    contrib = booster.predict(dm, pred_contribs=True)[0]
    cutoff = round(st.session_state.get("cutoff", AGGRESSIVE), 2)
    with right:
        approved = score < cutoff
        st.metric("Risk score", f"{score:.3f}", f"{'Approve' if approved else 'Decline'} at cut-off {cutoff:.2f}",
                  delta_color="green" if approved else "red", delta_arrow="off")
        s = pd.Series(contrib[:-1], index=FEATURES)
        s = s.reindex(s.abs().sort_values(ascending=False).index)
        shown, rest = s.head(8), s.iloc[8:].sum()
        wf = go.Figure(go.Waterfall(orientation="h", y=[describe(f) for f in shown.index] + ["All other features"],
                                    x=shown.tolist() + [rest], base=float(contrib[-1]),
                                    increasing=dict(marker_color=RED), decreasing=dict(marker_color=BLUE)))
        wf.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                         xaxis_title="Contribution to risk (log-odds)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(wf, width="stretch")
        st.caption(f"Bars start at the model's baseline ({contrib[-1]:.2f}). Red pushes risk up, blue pulls it down "
                   "(SHAP values computed by XGBoost). The score ranks risk; it runs higher than a true probability "
                   "because defaults were weighted 5× in training.")

# ---------- tab 3: model ----------
with tab_model:
    a, b = st.columns(2)
    with a:
        st.subheader("Ranks risk the same way on new customers")
        ro = T["rank_order"]
        fig = go.Figure()
        for smp, col in zip(SAMPLES, [GREY, BLUE, NAVY]):
            d = ro[ro["sample"] == smp]
            fig.add_bar(x=d["bin"], y=d["default_rate"], name=smp, marker_color=col)
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                          xaxis=dict(title="Score decile (1 = safest)", dtick=1),
                          yaxis=dict(title="Actual default rate", tickformat=".0%"), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        d2 = ro[ro["sample"] == "Test 2"]
        st.caption(f"On Test 2 the default rate climbs from {pct(d2['default_rate'].iloc[0])} in the safest tenth to "
                   f"{pct(d2['default_rate'].iloc[-1])} in the riskiest, with no reversals.")
    with b:
        st.subheader("XGBoost vs. neural network")
        comp = T["model_comparison"]
        fig = go.Figure()
        for _, r in comp.iterrows():
            fig.add_bar(x=["Train", "Test 1", "Test 2"], y=[r["AUC Train"], r["AUC Test 1"], r["AUC Test 2"]],
                        name=r["Model"], marker_color=BLUE if r["Model"] == "XGBoost" else GREY,
                        text=[f"{v:.3f}" for v in [r["AUC Train"], r["AUC Test 1"], r["AUC Test 2"]]], textposition="outside")
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                          yaxis=dict(title="AUC", range=[0.85, 0.95]), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption("XGBoost wins on every sample, handles missing values on its own, and has exact SHAP explanations. "
                   "The network is slightly more stable (smaller Train–Test gap).")

    st.subheader("What drives the score")
    imp = T["shap_importance"].head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=[describe(f) for f in imp["feature"]], orientation="h",
                           marker_color=BLUE))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis_title="Average impact on the score (mean |SHAP|, Test 2)")
    st.plotly_chart(fig, width="stretch")
    i1, i2 = st.columns(2)
    i1.image(str(DATA / "images" / "shap_beeswarm.png"), caption="SHAP beeswarm, Test 2: red = high value, right = more risk")
    i2.image(str(DATA / "images" / "shap_waterfall.png"), caption="One Test 2 customer scored 98.8%, explained feature by feature")

    st.subheader("How the model was chosen")
    g1, g2 = st.columns(2)
    for col, title, key in [(g1, "XGBoost: 72 models", "grid_xgb"), (g2, "Neural network: 32 models", "grid_nn")]:
        gdf = T[key].copy()
        gdf["Avg AUC"] = gdf[["AUC Train", "AUC Test 1", "AUC Test 2"]].mean(axis=1)
        gdf["Gap"] = gdf["AUC Train"] - gdf["AUC Test 2"]
        okm = gdf[gdf["Gap"] <= 0.02]
        best = okm["Avg AUC"].idxmax() if len(okm) else gdf["Gap"].idxmin()
        gdf = gdf[gdf["AUC Test 2"] > 0.7]  # hide the two networks that never learned (AUC ≈ 0.5)
        lo_, hi_ = gdf[["AUC Train", "AUC Test 2"]].min().min(), gdf[["AUC Train", "AUC Test 2"]].max().max()
        fig = go.Figure()
        fig.add_scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", line=dict(color=GREY, dash="dash"), name="Train = Test 2")
        fig.add_scatter(x=gdf["AUC Train"], y=gdf["AUC Test 2"], mode="markers",
                        marker=dict(color=BLUE, size=8, opacity=0.6), name="Model")
        fig.add_scatter(x=[T[key].loc[best, "AUC Train"]], y=[T[key].loc[best, "AUC Test 2"]], mode="markers",
                        marker=dict(color=ORANGE, size=18, symbol="star"), name="Chosen")
        fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                          xaxis_title="AUC Train", yaxis_title="AUC Test 2", legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(fig, width="stretch")
    st.caption("Rule: keep models whose Train and Test 2 AUC are within 0.02 (not memorizing), then take the highest "
               "average AUC. Dots far right of the dashed line overfit.")

# ---------- tab 4: about ----------
with tab_about:
    st.markdown(f"""
### The question
A card issuer wants to approve as many applicants as possible while keeping the default rate under a limit.
A model that ranks risk well reduces that to one decision: **where to put the cut-off.**

### Pipeline
1. **Data:** American Express Default Prediction (Kaggle). A random 20% of customers ({META['n_customers']:,})
   and 10% of the 188 statement features, read in chunks from the 16 GB file.
2. **Encoding:** one-hot encoding of categorical features, missing values kept as their own level.
3. **Feature engineering:** 339 account-level features from up to 13 monthly statements (averages, min/max,
   sums over 3 to 12 months, % change, latest value, response rates).
4. **Feature selection:** two XGBoost models, features above 0.5% importance in either → {len(FEATURES)}.
5. **Model selection:** 72 XGBoost and 32 neural-network models compared on Train, Test 1 and Test 2.
6. **Strategy:** cut-offs that keep the default rate under the limit on all three samples.

### Limitations and next steps
- **Revenue isn't profit:** defaulters earn 0, but what they don't repay isn't subtracted yet.
- **Scaled data:** amounts compare strategies but aren't dollars.
- **No later time period:** both test samples come from the same months and also guided choices.
- **Feature sampling:** only 10% of features were used, so no Payment features made it in.

### Data note
The competition data can't be redistributed, so this app ships only the trained model and summary
tables. The notebook in the repository rebuilds everything from the Kaggle files.
""")
