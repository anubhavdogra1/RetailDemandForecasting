"""Load the UCI Online Retail II dataset and turn it into a weekly demand panel."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "extracted" / "online_retail_II.xlsx"
CACHE = ROOT / "data" / "processed" / "transactions.parquet"
PANEL = ROOT / "data" / "processed" / "weekly_panel.parquet"


def load_transactions() -> pd.DataFrame:
    """Read both sheets, clean, and cache as parquet."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    sheets = pd.read_excel(XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["Description"] = df["Description"].astype("string")
    # Drop cancellations (invoice starts with C), non-positive qty/price, and non-product codes
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["StockCode"].str.match(r"^\d{5}")]
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df["CustomerID"] = df["CustomerID"].astype("string")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def build_panel(top_n: int = 50) -> pd.DataFrame:
    """Weekly units per SKU for the top_n SKUs by revenue, zero-filled.

    The first and last (partial) weeks are dropped.
    """
    df = load_transactions()
    top = df.groupby("StockCode")["Revenue"].sum().nlargest(top_n).index
    d = df[df["StockCode"].isin(top)]
    desc = d.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    weekly = d.groupby(["StockCode", "Week"]).agg(units=("Quantity", "sum"), revenue=("Revenue", "sum"),
                                                  price=("Price", "median"), orders=("Invoice", "nunique"),
                                                  customers=("CustomerID", "nunique")).reset_index()
    weeks = pd.date_range(df["Week"].min() + pd.Timedelta(weeks=1), df["Week"].max() - pd.Timedelta(weeks=1), freq="W-MON")
    idx = pd.MultiIndex.from_product([top, weeks], names=["StockCode", "Week"])
    panel = weekly.set_index(["StockCode", "Week"]).reindex(idx)
    panel[["units", "revenue", "orders", "customers"]] = panel[["units", "revenue", "orders", "customers"]].fillna(0)
    panel["price"] = panel.groupby(level=0)["price"].transform(lambda s: s.ffill().bfill())
    panel = panel.reset_index()
    panel["Description"] = panel["StockCode"].map(desc)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL)
    return panel


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL) if PANEL.exists() else build_panel()
