from server.db import get_db_connection

def get_active_forecasting_methods():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM forecasting_methods WHERE is_actively = true")

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [row[0] for row in rows]

def get_active_indicators():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM indicators_names WHERE is_actively = true")

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [row[0] for row in rows]
