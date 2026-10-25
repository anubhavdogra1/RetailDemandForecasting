"""Features, models and rolling-origin backtest for weekly SKU demand."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 4
LAGS = [1, 2, 3, 4, 8, 52]


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of week t, used to predict units in week t+h."""
    df = panel.sort_values(["StockCode", "Week"]).copy()
    g = df.groupby("StockCode")["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag - 1)  # lag_1 = this week's units (known at origin)
    for w in (4, 8, 13):
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["roll_std_8"] = g.transform(lambda s: s.rolling(8, min_periods=2).std())
    go = df.groupby("StockCode")["orders"]
    df["orders_roll_4"] = go.transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["cust_roll_4"] = df.groupby("StockCode")["customers"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["avg_order_size_8"] = df["roll_mean_8"] / df["orders_roll_4"].clip(lower=1)
    # market-wide momentum: total units across all SKUs, 4w vs 13w
    tot = df.groupby("Week")["units"].sum()
    df["mkt_ratio"] = df["Week"].map(tot.rolling(4, min_periods=1).mean() / tot.rolling(13, min_periods=1).mean())
    df["zero_share_13"] = g.transform(lambda s: (s == 0).rolling(13, min_periods=1).mean())
    df["scale"] = df["roll_mean_8"] + 1
    df["price_chg"] = df["price"] / df.groupby("StockCode")["price"].shift(4) - 1
    df["weekofyear"] = df["Week"].dt.isocalendar().week.astype(int)
    df["sku"] = df["StockCode"].astype("category")
    for h in range(1, HORIZON + 1):
        df[f"y_{h}"] = g.shift(-h)
    return df


FEATURES = [f"lag_{l}" for l in LAGS] + ["roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_std_8",
                                        "orders_roll_4", "cust_roll_4", "avg_order_size_8", "mkt_ratio",
                                        "zero_share_13", "price_chg", "weekofyear", "sku"]


def seasonal_naive(df: pd.DataFrame, h: int) -> pd.Series:
    """Same week last year if available, else last 4-week average."""
    return df["lag_52"].where(df["lag_52"].notna(), df["roll_mean_4"])


def moving_average(df: pd.DataFrame, h: int) -> pd.Series:
    return df["roll_mean_4"]


def fit_predict_gbm(train: pd.DataFrame, test: pd.DataFrame, h: int, quantile: float = 0.5) -> np.ndarray:
    """Predict demand relative to each SKU's recent level (scale-free, so one pooled model fits all SKUs)."""
    tr = train.dropna(subset=[f"y_{h}"])
    # short histories (e.g. no lag_52 yet) leave all-empty or constant columns, which sklearn's binning rejects
    cols = [c for c in FEATURES if tr[c].nunique(dropna=True) > 1]
    model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=300, learning_rate=0.03,
                                          max_leaf_nodes=8, min_samples_leaf=30, categorical_features="from_dtype",
                                          random_state=0)
    model.fit(tr[cols], (tr[f"y_{h}"] / tr["scale"]).clip(upper=20))
    return np.clip(model.predict(test[cols]), 0, None) * test["scale"].to_numpy()


def wape(y, yhat) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def predict_folds(feat: pd.DataFrame, n_folds: int = 8, step: int = 4) -> pd.DataFrame:
    """Rolling-origin backtest. Returns one row per (origin, SKU, horizon) with actuals and each model's forecast.

    Training rows for horizon h only use targets observed on or before the origin.
    """
    weeks = np.sort(feat["Week"].unique())
    last_origin = len(weeks) - 1 - HORIZON
    origins = [weeks[last_origin - i * step] for i in range(n_folds)][::-1]
    out = []
    for origin in origins:
        for h in range(1, HORIZON + 1):
            cutoff = origin - np.timedelta64(7 * h, "D")
            train = feat[feat["Week"] <= cutoff]
            test = feat[feat["Week"] == origin].dropna(subset=[f"y_{h}"]).copy()
            res = test[["StockCode", "Week", "scale", "roll_std_8"]].assign(origin=origin, h=h, y=test[f"y_{h}"])
            res["seasonal_naive"] = seasonal_naive(test, h)
            res["moving_avg_4"] = moving_average(test, h)
            res["gbm"] = fit_predict_gbm(train, test, h, 0.5)
            res["gbm_q90"] = fit_predict_gbm(train, test, h, 0.9)
            out.append(res)
    return pd.concat(out, ignore_index=True)


MODELS = ["seasonal_naive", "moving_avg_4", "gbm"]


def score(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (origin, h), g in preds.groupby(["origin", "h"]):
        for m in MODELS:
            rows.append(dict(origin=origin, h=h, model=m, wape=wape(g["y"], g[m]),
                             bias=(g[m].sum() - g["y"].sum()) / g["y"].sum()))
    return pd.DataFrame(rows)
