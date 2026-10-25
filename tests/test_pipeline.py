import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explain
import forecast as f
import inventory as inv


def toy_panel(weeks=80, skus=3, seed=0):
    rng = np.random.default_rng(seed)
    w = pd.date_range("2010-01-04", periods=weeks, freq="W-MON")
    df = pd.MultiIndex.from_product([[f"S{i}" for i in range(skus)], w], names=["StockCode", "Week"]).to_frame(index=False)
    df["units"] = rng.poisson(20, len(df)).astype(float)
    df["revenue"] = df["units"] * 2
    df["orders"] = rng.poisson(5, len(df)).astype(float)
    df["customers"] = df["orders"]
    df["price"] = 2.0 + rng.integers(0, 3, len(df)) * 0.5
    df["Description"] = "toy"
    return df


def test_features_do_not_leak_future():
    base = toy_panel()
    a = f.make_features(base)
    changed = base.copy()
    cut = base["Week"].iloc[40]
    changed.loc[changed["Week"] > cut, "units"] += 1000  # alter only the future
    b = f.make_features(changed)
    cols = [c for c in f.FEATURES if c != "sku"]
    ma, mb = a[a["Week"] <= cut], b[b["Week"] <= cut]
    pd.testing.assert_frame_equal(ma[cols].reset_index(drop=True), mb[cols].reset_index(drop=True))


def test_targets_are_future_units():
    feat = f.make_features(toy_panel())
    s0 = feat[feat.StockCode == "S0"].reset_index(drop=True)
    assert s0.loc[10, "y_2"] == s0.loc[12, "units"]


def test_backtest_training_uses_only_observed_targets():
    feat = f.make_features(toy_panel(weeks=60))
    preds = f.predict_folds(feat, n_folds=2, step=4)
    assert set(preds["h"]) == {1, 2, 3, 4}
    assert (preds["gbm_q90"] >= 0).all() and preds["y"].notna().all()
    # every forecast week is after its origin
    assert ((preds["Week"] == preds["origin"])).all()


def test_wape():
    assert f.wape(np.array([10, 10]), np.array([5, 15])) == 0.5


def test_inventory_fill_rate_and_overstock():
    p = pd.DataFrame({"y": [10.0, 10.0], "stock": [20.0, 5.0]})
    r = inv.evaluate(p, ["stock"]).iloc[0]
    assert r["fill_rate"] == 15 / 20
    assert r["overstock_units"] == 10


def test_match_fill_rate_hits_target():
    rng = np.random.default_rng(1)
    p = pd.DataFrame({"y": rng.poisson(20, 500).astype(float), "moving_avg_4": 20.0, "roll_std_8": 5.0})
    z, q = inv.match_fill_rate(p, 0.95)
    assert abs(inv.evaluate(q, ["moving_avg_matched"]).iloc[0]["fill_rate"] - 0.95) < 1e-3


def test_explain_fallback_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    facts = dict(product="S0 toy", avg_units_last_4_weeks=30.0, avg_units_last_13_weeks=20.0,
                 horizon_weeks=2, median_forecast=25.0, high_forecast_p90=40.0)
    assert "above" in explain.explain(facts)
