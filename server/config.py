import os

class Config:
    # Настройки безопасности
    SECRET_KEY = os.getenv("SECRET_KEY", "secret_key")
    JWT_EXPIRATION = 3600 # 1 час

    DB_NAME = "market_data"
    DB_USER = "postgres"
    DB_PASSWORD = "Kry1atsk032647"
    DB_HOST = "localhost"
    DB_PORT = 5432

    @staticmethod
    def get_db_url():
        return f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"