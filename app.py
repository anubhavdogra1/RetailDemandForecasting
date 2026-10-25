import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import data
import forecast as f
import explain
import inventory as inv

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Demand forecasting & stock planning")
st.caption("UCI Online Retail II · weekly demand for the top 50 products · 8-fold rolling backtest")


@st.cache_data
def load():
    panel = data.load_panel()
    preds = pd.read_parquet(data.ROOT / "data" / "processed" / "backtest_preds.parquet")
    return panel, preds


panel, preds = load()
tab1, tab2, tab3 = st.tabs(["Forecast accuracy", "Stock policy", "Product view"])

with tab1:
    s = f.score(preds)
    st.subheader("WAPE by model (lower is better)")
    st.bar_chart(s.groupby("model")["wape"].mean())
    st.dataframe(s.pivot_table(index="h", columns="model", values="wape").round(3), width='stretch')

with tab2:
    st.write("Stock for week t+2 is set at week t. The moving-average policy adds safety stock `z × recent std`; "
             "the model policy stocks to the 90th-percentile forecast.")
    p = inv.policy_levels(preds)
    target = inv.evaluate(p, ["gbm_q90_stock"]).iloc[0]["fill_rate"]
    z = st.slider("Safety factor z for moving-average policy", 0.0, 4.0, 1.64, 0.05)
    p = p.assign(moving_avg_ss=p["moving_avg_4"] + z * p["roll_std_8"].fillna(0))
    res = inv.evaluate(p, ["moving_avg_ss", "gbm_q90_stock"])
    st.dataframe(res.style.format({"fill_rate": "{:.1%}", "overstock_units": "{:,.0f}",
                                   "overstock_pct_of_demand": "{:.1%}", "sku_weeks_stocked_out": "{:.1%}"}),
                 width='stretch')
    st.info(f"Model policy fill rate: {target:.1%}. Move the slider until the baseline reaches the same fill rate "
            "and compare overstock.")

with tab3:
    sku = st.selectbox("Product", panel["StockCode"].unique(),
                       format_func=lambda c: f"{c} · {panel.loc[panel.StockCode == c, 'Description'].iat[0]}")
    h = st.radio("Horizon (weeks ahead)", [1, 2, 3, 4], index=1, horizontal=True)
    hist = panel[panel.StockCode == sku].set_index("Week")["units"].rename("actual")
    fc = preds[(preds.StockCode == sku) & (preds.h == h)].set_index("Week")[["gbm", "gbm_q90", "moving_avg_4"]]
    st.line_chart(pd.concat([hist, fc], axis=1))

    st.subheader("Ask about this forecast")
    origins = sorted(preds.loc[(preds.StockCode == sku) & (preds.h == h), "origin"].unique())
    if origins:
        origin = st.selectbox("Forecast made at week", origins, index=len(origins) - 1)
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
