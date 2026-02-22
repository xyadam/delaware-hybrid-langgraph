###########################################################################
##                            IMPORTS
###########################################################################

import pytest

from tools_sql import query_sql


###########################################################################
##                            TESTS
###########################################################################


def test_select_count_returns_data():
    result = query_sql.invoke({"sql": "SELECT COUNT(*) as cnt FROM products"})
    assert "200" in result


def test_non_select_rejected():
    result = query_sql.invoke({"sql": "INSERT INTO products VALUES (1, 'test')"})
    assert "only select" in result.lower()

    result = query_sql.invoke({"sql": "DELETE FROM products WHERE product_id = 1"})
    assert "only select" in result.lower()

    result = query_sql.invoke({"sql": "UPDATE products SET category = 'x'"})
    assert "only select" in result.lower()


def test_join_query_returns_results():
    sql = """
        SELECT p.description_en, SUM(t.quantity) as total_qty
        FROM transactions t
        JOIN products p ON t.product_id = p.product_id
        GROUP BY p.description_en
        ORDER BY total_qty DESC
        LIMIT 5
    """
    result = query_sql.invoke({"sql": sql})
    assert "description_en" in result
    assert "total_qty" in result
    lines = result.strip().split("\n")
    assert len(lines) >= 3  # header + separator + at least 1 data row


def test_empty_result():
    result = query_sql.invoke({"sql": "SELECT * FROM products WHERE product_id = -999"})
    assert "0 rows" in result.lower()


def test_invalid_sql_raises_exception():
    with pytest.raises(Exception, match="no such table"):
        query_sql.invoke({"sql": "SELECT * FROM nonexistent_table_xyz"})


def test_store_countries_present():
    result = query_sql.invoke({"sql": "SELECT DISTINCT country FROM stores ORDER BY country"})
    for country in ["China", "France", "Germany", "Portugal", "Spain", "United Kingdom", "United States"]:
        assert country in result


def test_transactions_are_2024_only():
    result = query_sql.invoke({"sql": "SELECT MIN(date) as min_d, MAX(date) as max_d FROM transactions"})
    assert "2024" in result
    assert "2023" not in result
    assert "2025" not in result
