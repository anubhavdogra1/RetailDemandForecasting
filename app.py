import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import data
import forecast as f
import explain
import inventory as inv

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting")
st.caption("UCI Online Retail II · weekly demand for the top 50 products · 8-fold rolling backtest")


@st.cache_data
def load():
    panel = data.load_panel()
    preds = pd.read_parquet(data.ROOT / "data" / "processed" / "backtest_preds.parquet")
    return panel, preds


MODEL_NAMES = {"gbm": "Model (median)", "moving_avg_4": "4-week moving average", "seasonal_naive": "Same week last year"}
POLICY_NAMES = {"moving_avg_ss": "Moving average + safety stock", "gbm_q90_stock": "Model 90th-percentile forecast"}

panel, preds = load()
tab1, tab2, tab3 = st.tabs(["Forecast accuracy", "Stock policy", "Product view"])

with tab1:
    s = f.score(preds).assign(model=lambda d: d["model"].map(MODEL_NAMES))
    st.subheader("Forecast error by model (WAPE, lower is better)")
    st.bar_chart(s.groupby("model")["wape"].mean().sort_values(), horizontal=True)
    by_h = s.pivot_table(index="h", columns="model", values="wape").round(3)
    by_h.index.name = "Weeks ahead"
    st.dataframe(by_h, width="stretch")

with tab2:
    st.write("Stock for week t+2 is set at week t. The moving-average policy adds safety stock `z × recent std`; "
             "the model policy stocks to the 90th-percentile forecast.")
    p = inv.policy_levels(preds)
    target = inv.evaluate(p, ["gbm_q90_stock"]).iloc[0]["fill_rate"]
    z = st.slider("Safety factor z for moving-average policy", 0.0, 4.0, 1.64, 0.05)
    p = p.assign(moving_avg_ss=p["moving_avg_4"] + z * p["roll_std_8"].fillna(0))
    res = inv.evaluate(p, ["moving_avg_ss", "gbm_q90_stock"])
    base, mdl = res.iloc[0], res.iloc[1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fill rate: baseline", f"{base['fill_rate']:.1%}")
    c2.metric("Fill rate: model", f"{mdl['fill_rate']:.1%}")
    c3.metric("Unsold units: baseline", f"{base['overstock_units']:,.0f}")
    c4.metric("Unsold units: model", f"{mdl['overstock_units']:,.0f}",
              f"{mdl['overstock_units'] / base['overstock_units'] - 1:+.1%} vs baseline", delta_color="inverse")
    show = res.assign(policy=res["policy"].map(POLICY_NAMES)).rename(columns={
        "policy": "Policy", "fill_rate": "Fill rate", "overstock_units": "Unsold units",
        "overstock_pct_of_demand": "Unsold as % of demand", "sku_weeks_stocked_out": "Stockout rate (SKU-weeks)"})
    st.dataframe(show.set_index("Policy").style.format({"Fill rate": "{:.1%}", "Unsold units": "{:,.0f}",
                                                        "Unsold as % of demand": "{:.1%}",
                                                        "Stockout rate (SKU-weeks)": "{:.1%}"}), width="stretch")
    st.info(f"The model policy's fill rate is {target:.1%}. Move the slider until the baseline reaches the same fill "
            "rate, then compare unsold units: that is the fair comparison (default 1.64 is about the match).")

with tab3:
    sku = st.selectbox("Product", panel["StockCode"].unique(),
                       format_func=lambda c: f"{c} · {panel.loc[panel.StockCode == c, 'Description'].iat[0]}")
    h = st.radio("Horizon (weeks ahead)", [1, 2, 3, 4], index=1, horizontal=True)
    import altair as alt

    hist = panel[panel.StockCode == sku][["Week", "units"]].assign(series="Actual")
    fc = preds[(preds.StockCode == sku) & (preds.h == h)].copy()
    fc["Week"] = fc["origin"] + pd.to_timedelta(7 * h, unit="D")  # the week being predicted, not the week it was made
    labels = {"gbm": "Model median", "gbm_q90": "Model 90th percentile", "moving_avg_4": "4-week moving average"}
    fc = fc.melt(id_vars="Week", value_vars=list(labels), var_name="series", value_name="units")
    fc["series"] = fc["series"].map(labels)
    order = ["Actual", *labels.values()]
    color = alt.Color("series:N", sort=order, title=None,
                      scale=alt.Scale(domain=order, range=["#4c78a8", "#59a14f", "#e15759", "#f2a900"]))
    line = alt.Chart(hist).mark_line().encode(x=alt.X("Week:T", title=None), y=alt.Y("units:Q", title="Units per week"),
                                               color=color)
    pts = alt.Chart(fc).mark_point(filled=True, size=80).encode(x="Week:T", y="units:Q", color=color,
                                                                tooltip=["series", "Week:T", alt.Tooltip("units:Q", format=",.0f")])
    st.altair_chart(line + pts, width="stretch")
    st.caption("Dots are backtest forecasts, plotted at the week they predicted (made " f"{h} week(s) earlier).")

    st.subheader("Ask about this forecast")
    origins = sorted(preds.loc[(preds.StockCode == sku) & (preds.h == h), "origin"].unique())
    if origins:
        origin = st.selectbox("Forecast made at week", origins, index=len(origins) - 1,
                              format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
        q = st.text_input("Question", "Explain this forecast and recommend how much to stock.")
        MAX_EXPLAINS = 5  # per session, so a public deployment can't burn through the API budget
        used = st.session_state.get("explains", 0)
        if st.button("Explain", disabled=used >= MAX_EXPLAINS):
            st.session_state["explains"] = used + 1
            feat = f.make_features(panel)
            with st.spinner("Thinking..."):
                st.write(explain.explain(explain.build_facts(feat, preds, sku, origin, h), q))
        if used >= MAX_EXPLAINS:
            st.caption("Explanation limit reached for this session.")
