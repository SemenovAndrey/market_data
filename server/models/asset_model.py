from server.db import get_db_connection
import psycopg2.extras
from datetime import datetime

def get_all_assets():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM assets ORDER BY symbol")

    assets = cur.fetchall()

    cur.close()
    conn.close()

    return assets

def get_asset_by_symbol(symbol):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM assets WHERE symbol = %s", (symbol,))

    asset = cur.fetchone()

    cur.close()
    conn.close()

    return asset

def update_asset_last_updated(symbol):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE assets SET last_updated = %s WHERE symbol = %s",
        (datetime.now(), symbol)
    )

    conn.commit()
    cur.close()
    conn.close()
