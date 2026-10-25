"""Turn forecasts into stocking decisions and compare policies on a simulated backtest.

Setup: lead time is 2 weeks, so stock for week t+2 is set at week t, one order-up-to level per SKU.
Fill rate = units served / units demanded. Overstock = units left unsold (holding cost proxy).
"""
import numpy as np
import pandas as pd

LEAD_H = 2


def policy_levels(preds: pd.DataFrame, z: float = 1.28) -> pd.DataFrame:
    p = preds[preds["h"] == LEAD_H].copy()
    p["moving_avg_ss"] = p["moving_avg_4"] + z * p["roll_std_8"].fillna(0)
    p["gbm_mean_only"] = p["gbm"]
    p["gbm_q90_stock"] = p["gbm_q90"]
    return p


def evaluate(p: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        served = np.minimum(p[c], p["y"])
        rows.append(dict(policy=c, fill_rate=served.sum() / p["y"].sum(),
                         overstock_units=(p[c] - served).sum(), overstock_pct_of_demand=(p[c] - served).sum() / p["y"].sum(),
                         sku_weeks_stocked_out=(p[c] < p["y"]).mean()))
    return pd.DataFrame(rows)


def match_fill_rate(p: pd.DataFrame, target: float) -> tuple[float, pd.DataFrame]:
    """Find the safety factor z for the moving-average policy that hits the target fill rate."""
    lo, hi = 0.0, 10.0
    for _ in range(40):
        z = (lo + hi) / 2
        lvl = p["moving_avg_4"] + z * p["roll_std_8"].fillna(0)
        fr = np.minimum(lvl, p["y"]).sum() / p["y"].sum()
        lo, hi = (z, hi) if fr < target else (lo, z)
    q = p.assign(moving_avg_matched=p["moving_avg_4"] + z * p["roll_std_8"].fillna(0))
    return z, q
