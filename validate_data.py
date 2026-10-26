
import sqlite3
from pathlib import Path

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")

EXPECTED_CITY_COUNTS = {
    "Bengaluru": 197,
    "Chennai": 191,
    "Delhi": 210,
    "Hyderabad": 191,
    "Mumbai": 184,
    "Pune": 227,
}

EXPECTED_GENDER_COUNTS = {
    "female": 610,
    "male": 560,
    "unknown": 30,
}


def load_queries(sql_text):
    """Load SQL statements, ignoring full-line comments."""
    statements = []
    current_lines = []

    for line in sql_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("--"):
            continue

        current_lines.append(line)

        if ";" in line:
            statement = "\n".join(current_lines).strip()
            if statement:
                statements.append(statement)
            current_lines = []

    remaining = "\n".join(current_lines).strip()
    if remaining:
        statements.append(remaining)

    return statements


def main():
    if not DB_PATH.exists():
        print(f"FAIL: Database not found: {DB_PATH}")
        return

    if not SQL_PATH.exists():
        print(f"FAIL: SQL file not found: {SQL_PATH}")
        return

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    results = []

    def check(description, condition, actual=None, expected=None):
        passed = bool(condition)
        results.append(passed)

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")

        if not passed:
            if expected is not None:
                print(f"       Expected: {expected}")
            if actual is not None:
                print(f"       Actual:   {actual}")

    try:
        print("\nPART A DATA VALIDATION")
        print("=" * 60)

        # Check required database tables.
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}

        for table in ["customers", "products", "orders"]:
            check(
                f"Table '{table}' exists",
                table in tables,
                actual=sorted(tables),
                expected=table,
            )

        if not {"customers", "products", "orders"}.issubset(tables):
            print("\nCannot continue: required tables are missing.")
            return

        # Row counts.
        cursor.execute("SELECT COUNT(*) FROM customers")
        customer_count = cursor.fetchone()[0]
        check(
            "Customer count",
            customer_count == 1200,
            customer_count,
            1200,
        )

        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]
        check(
            "Product count",
            product_count == 40,
            product_count,
            40,
        )

        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]
        check(
            "Clean order count",
            order_count == 12309,
            order_count,
            12309,
        )

        # Order status counts.
        cursor.execute("""
            SELECT status, COUNT(*) AS count
            FROM orders
            GROUP BY status
        """)
        status_counts = {
            row["status"]: row["count"]
            for row in cursor.fetchall()
        }

        check(
            "Completed order count",
            status_counts.get("completed", 0) == 11110,
            status_counts,
            "completed=11110",
        )

        check(
            "Cancelled order count",
            status_counts.get("cancelled", 0) == 1199,
            status_counts,
            "cancelled=1199",
        )

        check(
            "No pending orders remain",
            status_counts.get("pending", 0) == 0,
            status_counts,
            "pending=0",
        )

        # Customer city and gender counts.
        cursor.execute("""
            SELECT city, COUNT(*) AS count
            FROM customers
            GROUP BY city
        """)
        city_counts = {
            row["city"]: row["count"]
            for row in cursor.fetchall()
        }

        check(
            "Customer counts by city",
            city_counts == EXPECTED_CITY_COUNTS,
            city_counts,
            EXPECTED_CITY_COUNTS,
        )

        cursor.execute("""
            SELECT gender, COUNT(*) AS count
            FROM customers
            GROUP BY gender
        """)
        gender_counts = {
            row["gender"]: row["count"]
            for row in cursor.fetchall()
        }

        check(
            "Customer counts by gender",
            gender_counts == EXPECTED_GENDER_COUNTS,
            gender_counts,
            EXPECTED_GENDER_COUNTS,
        )

        # Age checks.
        cursor.execute("""
            SELECT
                SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END),
                MIN(age),
                MAX(age)
            FROM customers
        """)
        missing_ages, min_age, max_age = cursor.fetchone()

        check(
            "No missing customer ages",
            missing_ages == 0,
            missing_ages,
            0,
        )

        check(
            "All customer ages are between 18 and 60",
            min_age >= 18 and max_age <= 60,
            f"minimum={min_age}, maximum={max_age}",
            "minimum >= 18 and maximum <= 60",
        )

        # Order amounts and delivery times.
        cursor.execute("""
            SELECT
                SUM(CASE WHEN order_amount IS NULL THEN 1 ELSE 0 END),
                SUM(
                    CASE
                        WHEN status = 'completed'
                         AND delivery_minutes IS NULL
                        THEN 1 ELSE 0
                    END
                ),
                SUM(
                    CASE
                        WHEN status = 'cancelled'
                         AND delivery_minutes IS NULL
                        THEN 1 ELSE 0
                    END
                ),
                ROUND(SUM(
                    CASE
                        WHEN status = 'completed'
                        THEN order_amount ELSE 0
                    END
                ), 2)
            FROM orders
        """)
        missing_amounts, missing_completed_delivery, missing_cancelled_delivery, revenue = (
            cursor.fetchone()
        )

        check(
            "No missing order amounts",
            missing_amounts == 0,
            missing_amounts,
            0,
        )

        check(
            "No missing delivery times for completed orders",
            missing_completed_delivery == 0,
            missing_completed_delivery,
            0,
        )

        check(
            "Cancelled orders have missing delivery times",
            missing_cancelled_delivery == 1199,
            missing_cancelled_delivery,
            1199,
        )

        check(
            "Completed revenue equals INR 2,998,668.80",
            abs(revenue - 2998668.80) < 0.01,
            f"INR {revenue:,.2f}",
            "INR 2,998,668.80",
        )

        # Check required cleaned CSV files.
        for filename in [
            "customers.csv",
            "products.csv",
            "orders.csv",
        ]:
            path = Path("data/clean") / filename
            check(
                f"Clean CSV exists: {path}",
                path.exists(),
                actual=path.exists(),
                expected=True,
            )

        # Run all seven SQL queries.
        print("\nSQL QUERY VALIDATION")
        print("=" * 60)

        sql_text = SQL_PATH.read_text(encoding="utf-8")
        queries = load_queries(sql_text)

        check(
            "Exactly seven SQL queries are present",
            len(queries) == 7,
            len(queries),
            7,
        )

        if len(queries) == 7:
            expected_rows = {
                1: 30,
                2: 30,
                3: 11,
                4: 25,
                5: 550,
                # Q6 row count depends on customers with completed orders.
                7: 183,
            }

            query_rows = {}

            for number, query in enumerate(queries, start=1):
                try:
                    rows = connection.execute(query).fetchall()
                    query_rows[number] = rows

                    if number in expected_rows:
                        expected = expected_rows[number]
                        check(
                            f"Q{number} returns expected row count",
                            len(rows) == expected,
                            len(rows),
                            expected,
                        )
                    else:
                        check(
                            "Q6 executes successfully",
                            number == 6,
                            f"{len(rows)} rows",
                            "Successful execution",
                        )

                except sqlite3.Error as error:
                    check(
                        f"Q{number} executes successfully",
                        False,
                        str(error),
                        "No SQL error",
                    )

            # Verify revenue totals from Q2 and Q7.
            if 2 in query_rows and len(query_rows[2]) == 30:
                q2_total = round(
                    sum(row["completed_revenue"] for row in query_rows[2]),
                    2,
                )
                check(
                    "Q2 revenue total",
                    abs(q2_total - 2998668.80) < 0.01,
                    f"INR {q2_total:,.2f}",
                    "INR 2,998,668.80",
                )

                expected_largest_cell = 233909.45
                largest_cell = max(
                    row["completed_revenue"] for row in query_rows[2]
                )
                check(
                    "Q2 largest city/category revenue",
                    abs(largest_cell - expected_largest_cell) < 0.01,
                    f"INR {largest_cell:,.2f}",
                    "INR 233,909.45",
                )

            if 7 in query_rows and query_rows[7]:
                last_total = query_rows[7][-1]["cumulative_revenue"]
                check(
                    "Q7 final cumulative revenue",
                    abs(last_total - 2998668.80) < 0.01,
                    f"INR {last_total:,.2f}",
                    "INR 2,998,668.80",
                )

        print("\n" + "=" * 60)

        if all(results):
            print("ALL CHECKPOINTS PASSED")
        else:
            failed = len(results) - sum(results)
            print(f"VALIDATION INCOMPLETE: {failed} check(s) failed.")
            print("Review the FAIL messages above before proceeding.")

    except sqlite3.Error as error:
        print(f"\nDatabase validation error: {error}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
