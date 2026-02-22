###########################################################################
##                            IMPORTS
###########################################################################

import sqlite3
from pathlib import Path
from langchain.tools import tool


###########################################################################
##                           CONSTANTS
###########################################################################

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "sales.db"


###########################################################################
##                           SQL TOOL
###########################################################################

@tool
def query_sql(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the fashion retail sales database.

    The database contains: products, stores, customers, employees, discounts, transactions.
    All data is from 2024. Only SELECT statements are allowed.

    Args:
        sql: A SELECT SQL query to execute.

    Returns:
        Query results as formatted text with column headers, or an error message.
    """
    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed."

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(sql_stripped)
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return "Query returned 0 rows."

    columns = rows[0].keys()
    result_lines = [" | ".join(columns)]
    result_lines.append("-" * len(result_lines[0]))
    for row in rows[:200]:
        result_lines.append(" | ".join(str(row[col]) for col in columns))

    if len(rows) > 200:
        result_lines.append(f"... ({len(rows)} total rows, showing first 200)")

    conn.close()
    return "\n".join(result_lines)
