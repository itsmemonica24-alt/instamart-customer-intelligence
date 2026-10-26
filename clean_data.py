
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


def blank_values(series):
    """Treat missing values and whitespace-only strings as blank."""
    return (
        series.isna()
        | series.astype("string").str.strip().eq("").fillna(False)
    )


def clean_customers():
    customers = pd.read_csv(RAW_DIR / "customers.csv")

    # Standardise city names by trimming spaces and using title case.
    customers["city"] = (
        customers["city"].astype("string").str.strip().str.title()
    )

    # Standardise gender spellings.
    # Unrecognised or blank values become "unknown".
    gender = (
        customers["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z]", "", regex=True)
    )

    male_values = {"m", "male", "man", "men"}
    female_values = {"f", "female", "woman", "women"}

    customers["gender"] = gender.map(
        lambda value: (
            "male" if value in male_values
            else "female" if value in female_values
            else "unknown"
        )
    )

    # Convert ages to numbers. Invalid and out-of-range ages become missing.
    age = pd.to_numeric(customers["age"], errors="coerce")
    invalid_age = age.notna() & ((age < 10) | (age > 100))
    age = age.mask(invalid_age)

    print("\nCUSTOMER CLEANING")
    print(f"Invalid ages changed to missing: {invalid_age.sum()}")

    # Fill missing ages using the median age of each customer's city.
    customers["age"] = age
    city_medians = customers.groupby("city")["age"].transform("median")
    missing_age_before = customers["age"].isna().sum()
    customers["age"] = customers["age"].fillna(city_medians)

    # Safety fallback if a city has no valid age values.
    overall_median = customers["age"].median()
    customers["age"] = customers["age"].fillna(overall_median)

    print(f"Missing ages filled: {missing_age_before}")

    # Convert signup_date to a datetime.
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    # Check the required city and gender results.
    print("\nCustomer counts by city:")
    print(customers["city"].value_counts().sort_index().to_string())

    print("\nCustomer counts by gender:")
    print(customers["gender"].value_counts().sort_index().to_string())

    print(f"\nMissing ages remaining: {customers['age'].isna().sum()}")

    return customers


def clean_products():
    # No separate product-cleaning transformation is specified
    # in the supplied brief, so preserve the original product records.
    products = pd.read_csv(RAW_DIR / "products.csv")
    return products


def clean_orders():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")

    print("\nORDER CLEANING")

    # 1. Convert order timestamps to datetime.
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    # 2. Remove exact duplicate rows.
    before = len(orders)
    orders = orders.drop_duplicates().copy()
    print(f"Exact duplicate rows removed: {before - len(orders)}")

    # 3. Keep the latest record for each order_id.
    before = len(orders)
    orders = orders.sort_values(
        "order_ts", na_position="first", kind="stable"
    )
    orders = orders.drop_duplicates(
        subset=["order_id"], keep="last"
    ).copy()
    print(f"Older duplicate order records removed: {before - len(orders)}")

    # Standardise status spelling.
    orders["status"] = (
        orders["status"].astype("string").str.strip().str.lower()
    )

    # Convert amount, quantity and discount to numeric.
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    # 4. Fill missing/unparseable amounts using product price.
    price_lookup = products[["product_id", "price_inr"]].copy()
    price_lookup["price_inr"] = pd.to_numeric(
        price_lookup["price_inr"], errors="coerce"
    )

    orders = orders.merge(
        price_lookup,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    missing_amount_before = orders["order_amount"].isna().sum()

    calculated_amount = (
        orders["price_inr"]
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)

    orders["order_amount"] = orders["order_amount"].fillna(
        calculated_amount
    )

    print(f"Missing amounts filled: {missing_amount_before}")

    # 5. Negative delivery times are invalid and become missing.
    negative_delivery = orders["delivery_minutes"] < 0
    print(f"Negative delivery times changed to missing: {negative_delivery.sum()}")

    orders.loc[negative_delivery, "delivery_minutes"] = float("nan")

    # Cancelled orders were not delivered; leave their delivery time missing.
    cancelled = orders["status"].eq("cancelled")
    orders.loc[cancelled, "delivery_minutes"] = float("nan")

    # Fill missing delivery times for completed orders using the
    # median for their product category among completed orders.
    completed = orders["status"].eq("completed")
    missing_completed_before = (
        completed & orders["delivery_minutes"].isna()
    ).sum()

    category_lookup = products[["product_id", "category"]].copy()

    orders = orders.merge(
        category_lookup,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    completed_delivery = orders.loc[
        completed & orders["delivery_minutes"].notna()
    ]

    category_medians = completed_delivery.groupby(
        "category"
    )["delivery_minutes"].median()

    fill_mask = completed & orders["delivery_minutes"].isna()
    orders.loc[fill_mask, "delivery_minutes"] = (
        orders.loc[fill_mask, "category"].map(category_medians)
    )

    print(
        "Missing delivery times filled for completed orders: "
        f"{missing_completed_before}"
    )

    # Remove helper columns and restore a simple row index.
    orders = orders.drop(columns=["price_inr", "category"])
    orders = orders.reset_index(drop=True)

    return orders


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    customers = clean_customers()
    products = clean_products()
    orders = clean_orders()

    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(f"Clean customers: {len(customers):,}")
    print(f"Clean products: {len(products):,}")
    print(f"Clean orders: {len(orders):,}")

    print("\nOrder status counts:")
    print(orders["status"].value_counts().sort_index().to_string())

    print(
        "\nMissing order amounts: "
        f"{orders['order_amount'].isna().sum()}"
    )

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")

    print(
        "Completed orders with missing delivery time: "
        f"{orders.loc[completed, 'delivery_minutes'].isna().sum()}"
    )
    print(
        "Cancelled orders with missing delivery time: "
        f"{orders.loc[cancelled, 'delivery_minutes'].isna().sum()}"
    )

    print(
        "\nCompleted order revenue: INR "
        f"{orders.loc[completed, 'order_amount'].sum():,.2f}"
    )

    print(f"\nCleaned files saved in: {CLEAN_DIR.resolve()}")


if __name__ == "__main__":
    main()
