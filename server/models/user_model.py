from server.db import get_db_connection
import datetime
import psycopg2.extras

def get_user_by_username(username):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))

    user = cur.fetchone()

    cur.close()
    conn.close()

    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))

    user = cur.fetchone()

    cur.close()
    conn.close()

    return user

def create_user(username, email, password_hash):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users(username, email, password, created_at, graph_type_id) VALUES(%s, %s, %s, %s, %s)",
            (username, email, password_hash, datetime.datetime.now(), 1)
    )

    user_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return user_id
