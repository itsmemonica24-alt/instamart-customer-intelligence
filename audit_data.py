from pathlib import Path
import pandas as pd

#Project Folder
RAW_DIR = Path("data/raw")

def load_csv(filename, keep_default_na=True):
    """Load a CSV file from the raw data folder."""
    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Run generate_data.py first."
        )
    return pd.read_csv(file_path, keep_default_na=keep_default_na)

def is_blank(series):
    """Identify missing values and strings containing only whitespace."""
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

def audit_table(name, dataframe):
    """Display a dataset's shape, column types, and missing values."""
    print("\n" + "=" * 70)
    print(f"DATASET AUDIT: {name}")
    print("=" * 70)

    print(f"\nShape: {dataframe.shape}")
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\nMissing values by column:")
    missing_count = dataframe.apply(lambda col: is_blank(col).sum())
    missing_percentage = (missing_count / len(dataframe) * 100).round(2)

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    print(missing_report.to_string())

def main():
    #Load the original raw datasets.
    customers = load_csv("customers.csv")
    products = load_csv("products.csv")
    orders = load_csv("orders.csv")

    #Audit each dataset.
    audit_table("customers.csv", customers)
    audit_table("products.csv", products)
    audit_table("orders.csv", orders)

    #----------------------------------------------------------
    #Specific data-quality checks required by the task
    #---------------------------------------------------------

    print("\n" + "=" * 70)
    print("REQUIRED DATA-QUALITY CHECKS")
    print("=" * 70)

    #Order Checks
    print("\nORDERS")
    print("_" * 40)

    print(f"Raw order rows: {len(orders)}")
    print(f"Exact duplicate rows {orders.duplicated().sum()}")

    if "status" in orders.columns:
        pending = (
            orders["status"]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("pending")
            .fillna(False)
        )
        print(f"Pending rows: {pending.sum()}")
    else:
        print("Pending rows: status column not found")

    if "order_id" in orders.columns:
        print(
            "Distinct order_id values:"
            f"{orders['order_id'].nunique(dropna=True)}"
        )
    else:
        print("Distinct order_id values: order_id column not found")

    if "order_amount" in orders.columns:
        amount_numeric = pd.to_numeric(
            orders["order_amount"], errors="coerce"
        )
        print(
            "order amounts that cannnot be converted to numbers:"
            f"{amount_numeric.isna().sum()}"
        )
    else:
        print("Order amounts: order_amount column not found")

    if "delivery_minutes" in orders.columns:
        delivery_numeric = pd.to_numeric(
            orders["delivery_minutes"], errors="coerce"
        )

        negative_delivery = delivery_numeric < 0
        print(f"Negative delivery times: {negative_delivery.sum()}")

        blank_delivery = is_blank(orders["delivery_minutes"])
        print(f"Blank delivery times: {blank_delivery.sum()}")
    else:
        print("Delivery times: delivery_minutes column not found")

    #Customer checks
    print("\nCUSTOMERS")
    print("-" * 40)

    if "age" in customers.columns:
        age_numeric = pd.to_numeric(customers["age"], errors="coerce")
        blank_age = is_blank(customers["age"])

        outside_age_range = (
            age_numeric.notna()
            & ((age_numeric < 10) | (age_numeric > 100))
        )

        print(f"Blank age values: {blank_age.sum()}")
        print(
            "Age values outside 10-100:"
            f"{outside_age_range.sum()}"
        )
    else:
        print("Age checks: age column not found")

    if "city" in customers.columns:
        nonblank_city = customers.loc[
            ~is_blank(customers["city"]), "city"
        ]
        print(
            "Distinct nonblank raw city spellings:"
            f"{nonblank_city.nunique()}"
        )
    else:
        print("City checks: city column not found")

    if "gender" in customers.columns:
        nonblank_gender = customers.loc[
            ~is_blank(customers["gender"]), "gender"
        ]
        blank_gender = is_blank(customers["gender"])

        print(
            "Distinct nonblank raw gender spellings:"
            f"{nonblank_gender.nunique()}"
        )
        print(f"Blank gender values: {blank_gender.sum()}")
    else:
        print("Gender checks: gender column not found")

    print("\nAudit completed. The raw datasets have not beeen modified.")

if __name__ == "__main__":
    main()