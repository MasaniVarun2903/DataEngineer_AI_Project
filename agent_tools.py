"""
STEP 3: The "tools" that the AI agent is allowed to use to answer questions.
These read the small aggregated CSV that Spark produced (fast - no need to
start a Spark session every time someone asks a question, including when
deployed on Streamlit Cloud where Spark isn't installed at all).
"""

import glob
import pandas as pd


def _load(folder: str) -> pd.DataFrame:
    file = glob.glob(f"{folder}/part-*.csv")[0]
    return pd.read_csv(file)


def get_sales_by_region(region: str, category: str = None) -> str:
    """Get total units and revenue for a region, optionally filtered by category."""
    df = _load("output/agg_region_category_month")
    result = df[df["region"].str.lower() == region.lower()]
    if category:
        result = result[result["category"].str.lower() == category.lower()]
    if result.empty:
        return f"No data found for region={region}, category={category}"
    total_units = int(result["total_units"].sum())
    total_revenue = result["total_revenue"].sum()
    return (
        f"Region: {region}, Category: {category or 'All'} -> "
        f"Total units sold: {total_units}, Total revenue: Rs.{total_revenue:,.2f}"
    )


def compare_month_over_month(region: str, category: str, year: int, month: int) -> str:
    """Compare a given month's sales to the previous month for a region+category."""
    df = _load("output/agg_region_category_month")
    df = df[
        (df["region"].str.lower() == region.lower())
        & (df["category"].str.lower() == category.lower())
    ]

    prev_month = month - 1 if month > 1 else 12
    prev_year = year if month > 1 else year - 1

    current = df[(df["year"] == year) & (df["month"] == month)]
    previous = df[(df["year"] == prev_year) & (df["month"] == prev_month)]

    if current.empty or previous.empty:
        return "Not enough data to compare these months."

    cur_units = int(current["total_units"].sum())
    prev_units = int(previous["total_units"].sum())
    change_pct = ((cur_units - prev_units) / prev_units) * 100 if prev_units else 0

    trend = "increased" if change_pct > 0 else "dropped"
    return (
        f"{category} sales in {region} {trend} by {abs(change_pct):.1f}% "
        f"in {year}-{month:02d} compared to the previous month "
        f"({prev_units} -> {cur_units} units)."
    )


def detect_big_drops(threshold_pct: float = 30.0) -> str:
    """Scan all region+category combos and flag month-over-month drops bigger than threshold_pct."""
    df = _load("output/agg_region_category_month")
    df = df.sort_values(["region", "category", "year", "month"])
    alerts = []

    for (region, category), group in df.groupby(["region", "category"]):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            prev_units = group.loc[i - 1, "total_units"]
            cur_units = group.loc[i, "total_units"]
            if prev_units > 0:
                change_pct = ((cur_units - prev_units) / prev_units) * 100
                if change_pct <= -threshold_pct:
                    alerts.append(
                        f"{category} in {region}: dropped {abs(change_pct):.1f}% "
                        f"in {int(group.loc[i, 'year'])}-{int(group.loc[i, 'month']):02d}"
                    )
    if not alerts:
        return "No major drops detected."
    return "Alerts found:\n" + "\n".join(alerts)