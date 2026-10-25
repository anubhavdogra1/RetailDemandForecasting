"""Grounded natural-language explanations of a forecast, written by Claude.

The model only sees numbers computed here; it is told not to invent causes.
"""
import os

import pandas as pd

MODEL = "claude-sonnet-5-5"

SYSTEM = (
    "You are a demand-planning assistant for a wholesale retailer. Answer using ONLY the facts provided. "
    "If the facts don't explain something, say so instead of guessing causes. Be concise (under 150 words), "
    "plain language, and give a concrete stocking recommendation when asked."
)


def build_facts(feat: pd.DataFrame, preds: pd.DataFrame, sku: str, origin, h: int) -> dict:
    """Numbers about one SKU at one forecast origin."""
    row = feat[(feat["StockCode"] == sku) & (feat["Week"] == origin)].iloc[0]
    p = preds[(preds["StockCode"] == sku) & (preds["origin"] == origin) & (preds["h"] == h)]
    facts = {
        "product": f"{sku} {row.get('Description', '')}".strip(),
        "forecast_origin_week": str(pd.Timestamp(origin).date()),
        "horizon_weeks": h,
        "units_last_week": float(row["lag_1"]),
        "avg_units_last_4_weeks": float(row["roll_mean_4"]),
        "avg_units_last_13_weeks": float(row["roll_mean_13"]),
        "weekly_std_last_8_weeks": None if pd.isna(row["roll_std_8"]) else float(row["roll_std_8"]),
        "share_of_zero_weeks_last_13": float(row["zero_share_13"]),
        "orders_per_week_last_4": float(row["orders_roll_4"]),
        "market_momentum_4w_vs_13w": float(row["mkt_ratio"]),
    }
    if len(p):
        p = p.iloc[0]
        facts.update(median_forecast=float(p["gbm"]), high_forecast_p90=float(p["gbm_q90"]),
                     baseline_moving_avg=float(p["moving_avg_4"]))
    return facts


def fallback(facts: dict) -> str:
    """Used when no API key is configured, so the app still works offline."""
    f = facts
    trend = "above" if f["avg_units_last_4_weeks"] > f["avg_units_last_13_weeks"] else "below"
    out = (f"{f['product']}: recent 4-week average of {f['avg_units_last_4_weeks']:.0f} units/week is {trend} "
           f"its 13-week average ({f['avg_units_last_13_weeks']:.0f}).")
    if "median_forecast" in f:
        out += (f" Expected demand in {f['horizon_weeks']} weeks is about {f['median_forecast']:.0f} units; "
                f"stocking {f['high_forecast_p90']:.0f} covers a high-demand week (90th percentile).")
    return out + " (Set ANTHROPIC_API_KEY for a full explanation.)"


def explain(facts: dict, question: str = "Explain this forecast and recommend how much to stock.") -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return fallback(facts)
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": f"Facts:\n{facts}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text
