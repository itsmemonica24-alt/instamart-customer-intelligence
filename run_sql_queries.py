
import sqlite3
from pathlib import Path

DB_PATH = Path("data/instamart.db")
SQL_PATH = Path("sql/queries.sql")


def load_queries(sql_text):
    """Split SQL into executable statements, ignoring comment-only sections."""
    statements = []
    current_lines = []

    for line in sql_text.splitlines():
        stripped = line.strip()

        # Ignore full-line SQL comments.
        if stripped.startswith("--"):
            continue

        current_lines.append(line)

        # A semicolon terminates a statement in this assignment's SQL file.
        if ";" in line:
            statement = "\n".join(current_lines).strip()
            if statement:
                statements.append(statement)
            current_lines = []

    # Handle a final statement without a trailing semicolon.
    remaining = "\n".join(current_lines).strip()
    if remaining:
        statements.append(remaining)

    return statements


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run python clean_data.py first."
        )

    if not SQL_PATH.exists():
        raise FileNotFoundError(
            f"SQL file not found: {SQL_PATH}\n"
            "Create sql/queries.sql first."
        )

    sql_text = SQL_PATH.read_text(encoding="utf-8")
    queries = load_queries(sql_text)

    print(f"Found {len(queries)} SQL statements.")

    if len(queries) != 7:
        print(
            "\nWARNING: Expected 7 queries. "
            "Please inspect sql/queries.sql for extra or missing statements."
        )
        for number, query in enumerate(queries, start=1):
            print(f"\n--- STATEMENT {number} ---")
            print(query[:300])
        return

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        for number, query in enumerate(queries, start=1):
            print("\n" + "=" * 70)
            print(f"QUERY {number}")
            print("=" * 70)

            cursor = connection.execute(query)
            rows = cursor.fetchall()

            column_names = [
                description[0] for description in cursor.description
            ]

            print("Columns:", ", ".join(column_names))
            print("Rows returned:", len(rows))

            for row in rows[:10]:
                print(dict(row))

            if len(rows) > 10:
                print(f"... and {len(rows) - 10} more rows.")

            # Additional checks for expected results.
            if number == 1:
                print("Expected rows: 30")
            elif number == 2:
                total = sum(row["completed_revenue"] for row in rows)
                print(f"Total completed revenue: INR {total:,.2f}")
                print("Expected rows: 30")
            elif number == 3:
                print("Expected rows: 11")
            elif number == 4:
                print("Expected rows: 25")
            elif number == 5:
                print("Expected rows: 550")
            elif number == 6:
                print(
                    "Check customer ranks and top-spending customers "
                    "for each city."
                )
            elif number == 7:
                if rows:
                    last_total = rows[-1]["cumulative_revenue"]
                    print(
                        f"Final cumulative revenue: INR {last_total:,.2f}"
                    )
                print("Expected rows: 183")

    except sqlite3.Error as error:
        print(f"\nSQL execution error: {error}")
        print("Check the query that was printed immediately before the error.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
