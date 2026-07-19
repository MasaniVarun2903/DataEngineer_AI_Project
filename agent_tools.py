"""
STEP 3: The "tools" that the AI agent is allowed to use to answer questions.
Some tools read the small aggregated CSV Spark produced (monthly summaries).
The newer weekly tools read directly from the raw weekly data
(retail_sales_raw.csv), since Spark's output here is only aggregated by month.
"""

import glob
import pandas as pd


def _load(folder: str) -> pd.DataFrame:
    file = glob.glob(f"{folder}/part-*.csv")[0]
    return pd.read_csv(file)


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv("retail_sales_raw.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _nearest_date(dates: pd.Series, target: str, tolerance_days: int = 3):
    """Find the closest available date to the requested one, within a few days.
    Our sample data is weekly, so exact date matches aren't always guaranteed."""
    target_dt = pd.to_datetime(target)
    unique_dates = dates.drop_duplicates()
    diffs = (unique_dates - target_dt).abs()
    if diffs.empty:
        return None
    closest_idx = diffs.idxmin()
    if diffs.loc[closest_idx] <= pd.Timedelta(days=tolerance_days):
        return unique_dates.loc[closest_idx]
    return None


# ---------------------------------------------------------------------------
# Existing tools (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NEW tools - all work at the WEEKLY level, reading the raw diary directly
# ---------------------------------------------------------------------------

def get_highest_selling_in_week(region: str, start_date: str, end_date: str, category: str = None) -> str:
    """Find the top-selling category for a region within a date range (e.g. one week),
    the top-selling PRODUCT WITHIN that same category, AND separately the single
    highest-selling product overall (which may belong to a different category).
    If a specific category is given, skips the "find top category" step and just
    reports the top product within that requested category instead."""
    df = _load_raw()
    mask = (
        (df["region"].str.lower() == region.lower())
        & (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    )
    subset = df[mask]
    if subset.empty:
        return f"No data found for {region} between {start_date} and {end_date}."

    if category:
        cat_subset = subset[subset["category"].str.lower() == category.lower()]
        if cat_subset.empty:
            return f"No data found for category={category} in {region} between {start_date} and {end_date}."
        by_product = cat_subset.groupby("product")["units_sold"].sum().sort_values(ascending=False)
        top_product, top_units = by_product.index[0], int(by_product.iloc[0])
        return (
            f"In the {category} category, in {region} between {start_date} and {end_date}, "
            f"the top-selling product was {top_product} ({top_units} units sold)."
        )

    by_category = subset.groupby("category")["units_sold"].sum().sort_values(ascending=False)
    top_category, top_category_units = by_category.index[0], int(by_category.iloc[0])

    within_top_category = subset[subset["category"] == top_category]
    by_product_in_cat = within_top_category.groupby("product")["units_sold"].sum().sort_values(ascending=False)
    top_product_in_cat, top_product_in_cat_units = by_product_in_cat.index[0], int(by_product_in_cat.iloc[0])

    by_product_overall = (
        subset.groupby(["product", "category"])["units_sold"].sum().sort_values(ascending=False)
    )
    overall_top_product, overall_top_category = by_product_overall.index[0]
    overall_top_units = int(by_product_overall.iloc[0])

    result = (
        f"Between {start_date} and {end_date} in {region}:\n"
        f"- Top-selling CATEGORY: {top_category} ({top_category_units} units sold)\n"
        f"- Top-selling PRODUCT within {top_category}: {top_product_in_cat} ({top_product_in_cat_units} units sold)\n"
        f"- Highest-selling PRODUCT overall (any category): {overall_top_product} "
        f"from the {overall_top_category} category ({overall_top_units} units sold)"
    )

    if overall_top_product == top_product_in_cat and overall_top_category == top_category:
        result += "\n(This is also the single best-selling product overall.)"

    return result



def compare_products_week_over_week(region: str, category: str, week_start_date: str) -> str:
    """Compare a category's units sold in a given week vs the previous week (7 days earlier) for a region."""
    df = _load_raw()
    filtered = df[
        (df["region"].str.lower() == region.lower())
        & (df["category"].str.lower() == category.lower())
    ]
    if filtered.empty:
        return f"No data found for {category} in {region}."

    target_date = _nearest_date(filtered["date"], week_start_date)
    if target_date is None:
        return f"No week close to {week_start_date} found in the data for {category} in {region}."

    prev_target = target_date - pd.Timedelta(days=7)
    prev_date = _nearest_date(filtered["date"], prev_target.strftime("%Y-%m-%d"), tolerance_days=2)
    if prev_date is None:
        return f"Found the week of {target_date.date()}, but no earlier week to compare against."

    current = filtered[filtered["date"] == target_date]
    previous = filtered[filtered["date"] == prev_date]

    cur_units = int(current["units_sold"].sum())
    prev_units = int(previous["units_sold"].sum())
    change_pct = ((cur_units - prev_units) / prev_units) * 100 if prev_units else 0
    trend = "increased" if change_pct > 0 else "decreased"

    return (
        f"{category} sales in {region} {trend} by {abs(change_pct):.1f}% "
        f"for the week of {target_date.date()} compared to the week of {prev_date.date()} "
        f"({prev_units} -> {cur_units} units)."
    )


def detect_declining_products(region: str = None, min_consecutive_weeks: int = 5) -> str:
    """Scan weekly sales per product and flag any product declining for several consecutive weeks in a row.
    Optionally filter to a single region."""
    df = _load_raw()
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if df.empty:
        return f"No data found for region={region}."

    alerts = []
    group_cols = ["product", "region"]
    for keys, group in df.groupby(group_cols):
        weekly = group.groupby("date", as_index=False)["units_sold"].sum().sort_values("date").reset_index(drop=True)
        streak = 0
        for i in range(1, len(weekly)):
            if weekly.loc[i, "units_sold"] < weekly.loc[i - 1, "units_sold"]:
                streak += 1
            else:
                streak = 0
            if streak >= min_consecutive_weeks - 1:
                product_name, region_name = keys
                last_date = weekly.loc[i, "date"].strftime("%Y-%m-%d")
                alerts.append(
                    f"{product_name} in {region_name}: declining for {streak + 1} "
                    f"consecutive weeks (as of week of {last_date})"
                )
                break  # only report the first time it crosses the threshold

    if not alerts:
        scope = f" in {region}" if region else ""
        return f"No product shows a consistent multi-week decline{scope} right now."
    return "Products with a consistent week-over-week decline:\n" + "\n".join(alerts)
