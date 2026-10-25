"""Rebuild everything: python scripts/run_all.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data
import forecast as f
import inventory as inv

panel = data.build_panel()
preds = f.predict_folds(f.make_features(panel))
preds.to_parquet(data.ROOT / "data" / "processed" / "backtest_preds.parquet")
s = f.score(preds)
print(s.groupby("model")[["wape", "bias"]].mean().round(3))
p = inv.policy_levels(preds)
target = inv.evaluate(p, ["gbm_q90_stock"]).iloc[0]["fill_rate"]
z, q = inv.match_fill_rate(p, target)
print(f"\nAt fill rate {target:.1%} (baseline z={z:.2f}):")
print(inv.evaluate(q, ["moving_avg_matched", "gbm_q90_stock"]).round(3).to_string(index=False))
