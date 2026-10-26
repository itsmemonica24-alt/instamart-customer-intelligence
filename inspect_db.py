import sqlite3

connection = sqlite3.connect("data/instamart.db")
cursor = connection.cursor()

tables = ["customers", "products", "orders"]

for table in tables:
    print(f"\n--- {table.upper()} ---")
    cursor.execute(f"PRAGMA table_info({table})")

    columns = cursor.fetchall()

    if not columns:
        print("Table not found or has no columns.")
    else:
        for column in columns:
            print(f"{column[1]} ({column[2]})")

connection.close()
