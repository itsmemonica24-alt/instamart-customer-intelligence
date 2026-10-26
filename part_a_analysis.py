
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # --------------------------------------------------------
    # 1. LOAD CLEANED DATASETS
    # --------------------------------------------------------

    print("=" * 65)
    print("PART A: PANDAS CROSS-CHECK AND EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    numeric_columns = [
        "order_amount",
        "quantity",
        "discount_pct",
        "delivery_minutes",
    ]

    for column in numeric_columns:
        orders[column] = pd.to_numeric(
            orders[column], errors="coerce"
        )

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    completed = orders.loc[
        orders["status"].eq("completed")
    ].copy()

    # --------------------------------------------------------
    # 2. PANDAS CROSS-CHECK USING A PIVOT TABLE
    # --------------------------------------------------------

    merged = (
        completed
        .merge(
            customers[["customer_id", "city"]],
            on="customer_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="inner",
            validate="many_to_one",
        )
    )

    pivot = pd.pivot_table(
        merged,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    ).round(2)

    pivot.to_csv(OUTPUT_DIR / "city_category_revenue_pivot.csv")

    pandas_revenue = round(
        float(completed["order_amount"].sum()), 2
    )
    pivot_total = round(float(pivot.loc["All", "All"]), 2)

    print("\n1. PANDAS CROSS-CHECK")
    print("-" * 65)
    print(
        f"Completed revenue from cleaned orders: "
        f"INR {pandas_revenue:,.2f}"
    )
    print(f"Grand total from pandas pivot: INR {pivot_total:,.2f}")

    if abs(pandas_revenue - 2998668.80) < 0.01:
        print("PASS: Completed revenue matches the reference.")
    else:
        print("FAIL: Completed revenue differs from the reference.")

    if abs(pivot_total - pandas_revenue) < 0.01:
        print("PASS: Pivot total matches completed revenue.")
    else:
        print("FAIL: Pivot total does not match completed revenue.")

    print("\nRevenue by city and category:")
    print(pivot.to_string())

    # --------------------------------------------------------
    # 3. GROUPBY AND NAMED AGGREGATION BY CITY
    # --------------------------------------------------------

    city_summary = (
        merged.groupby("city")
        .agg(
            completed_revenue=("order_amount", "sum"),
            mean_order_amount=("order_amount", "mean"),
            completed_order_count=("order_amount", "count"),
        )
        .round(2)
        .sort_index()
    )

    city_summary.to_csv(OUTPUT_DIR / "city_order_summary.csv")

    print("\n\n2. COMPLETED ORDERS BY CITY")
    print("-" * 65)
    print(city_summary.to_string())

    # --------------------------------------------------------
    # 4. DESCRIPTIVE STATISTICS AND OUTLIER ANALYSIS
    # --------------------------------------------------------

    amounts = completed["order_amount"].dropna()

    mean_amount = amounts.mean()
    median_amount = amounts.median()
    skewness = amounts.skew()

    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers_below = int((amounts < lower_fence).sum())
    outliers_above = int((amounts > upper_fence).sum())

    print("\n\n3. COMPLETED ORDER AMOUNT DISTRIBUTION")
    print("-" * 65)
    print(f"Completed orders analysed: {len(amounts):,}")
    print(f"Mean: INR {mean_amount:,.2f}")
    print(f"Median: INR {median_amount:,.2f}")
    print(f"Skewness: {skewness:.2f}")
    print(f"First quartile (Q1): INR {q1:,.2f}")
    print(f"Third quartile (Q3): INR {q3:,.2f}")
    print(f"IQR: INR {iqr:,.2f}")
    print(f"Lower outlier fence: INR {lower_fence:,.2f}")
    print(f"Upper outlier fence: INR {upper_fence:,.2f}")
    print(f"Outliers below lower fence: {outliers_below}")
    print(f"Outliers above upper fence: {outliers_above}")

    # --------------------------------------------------------
    # 5. CORRELATION ANALYSIS
    # --------------------------------------------------------

    correlation_columns = [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes",
    ]

    correlation_data = completed[correlation_columns]
    correlation_matrix = correlation_data.corr().round(4)

    correlation_matrix.to_csv(
        OUTPUT_DIR / "completed_order_correlations.csv"
    )

    print("\n\n4. CORRELATION MATRIX")
    print("-" * 65)
    print(correlation_matrix.to_string())

    # FIX: use pandas .loc assignment instead of modifying
    # the underlying NumPy array with np.fill_diagonal().
    absolute_corr = correlation_matrix.abs().copy()

    for column in absolute_corr.columns:
        absolute_corr.loc[column, column] = np.nan

    correlations = absolute_corr.stack()

    if not correlations.empty:
        strongest_pair = correlations.idxmax()

        strongest_value = correlation_matrix.loc[
            strongest_pair[0], strongest_pair[1]
        ]

        print(
            "\nStrongest absolute correlation:",
            f"{strongest_pair[0]} vs {strongest_pair[1]} = "
            f"{strongest_value:.4f}",
        )

    print(
        "Note: Correlation describes association, not causation."
    )

    # --------------------------------------------------------
    # 6. SAVE EDA REPORT
    # --------------------------------------------------------

    report_lines = [
        "PART A: EXPLORATORY DATA ANALYSIS",
        "",
        "Dataset overview:",
        (
            f"The cleaned dataset contains {len(customers):,} "
            f"customers, {len(products):,} products, and "
            f"{len(orders):,} orders."
        ),
        (
            f"There are {len(completed):,} completed orders "
            f"with total revenue of INR {pandas_revenue:,.2f}."
        ),
        "",
        "Completed order amount distribution:",
        f"Mean: INR {mean_amount:,.2f}",
        f"Median: INR {median_amount:,.2f}",
        f"Skewness: {skewness:.2f}",
        f"Q1: INR {q1:,.2f}",
        f"Q3: INR {q3:,.2f}",
        f"IQR: INR {iqr:,.2f}",
        f"Lower IQR fence: INR {lower_fence:,.2f}",
        f"Upper IQR fence: INR {upper_fence:,.2f}",
        f"Outliers below lower fence: {outliers_below}",
        f"Outliers above upper fence: {outliers_above}",
        "",
        "Correlation matrix:",
        correlation_matrix.to_string(),
        (
            "Correlations indicate associations and do not "
            "establish causal relationships."
        ),
        "",
        "Revenue by city:",
        city_summary.to_string(),
        "",
        "Revenue by city and category:",
        pivot.to_string(),
    ]

    report_path = OUTPUT_DIR / "part_a_eda_report.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 7. CONFIRM OUTPUT FILES
    # --------------------------------------------------------

    print("\n\n5. OUTPUT FILES")
    print("-" * 65)

    output_files = [
        "city_category_revenue_pivot.csv",
        "city_order_summary.csv",
        "completed_order_correlations.csv",
        "part_a_eda_report.txt",
    ]

    for filename in output_files:
        path = OUTPUT_DIR / filename
        print(f"{'Saved' if path.exists() else 'MISSING'}: {path}")

    print("\nPart A analysis finished.")


if __name__ == "__main__":
    main()
