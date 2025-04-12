import psycopg2

from server.config import Config

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
        )

        print('Connection established')
        return conn
    except Exception as e:
        print(e)
        return None
