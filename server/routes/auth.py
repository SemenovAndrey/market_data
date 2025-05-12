from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime

from server.config import Config
from server.db import get_db_connection

auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"Ошибка": "Логин и пароль обязательны для заполнения"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT password FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user[0], password):
            token = jwt.encode(
                {
                    "username": username,
                    "exp": datetime.datetime.now() + datetime.timedelta(seconds=Config.JWT_EXPIRATION),
                },
                Config.SECRET_KEY,
                algorithm="HS256"
            )

            return jsonify({"token": token}), 200
        else:
            return jsonify({"error": "Неверный логин или пароль"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@auth_blueprint.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({"error": "Все поля обязательны для заполнения"}), 400

    if not "@" in email:
        return jsonify({"error": "Почта указана некорректно"}), 400

    if len(email.split('@')) > 2:
        return jsonify({"error": "Почта указана некорректно"}), 400

    if not "." in email.split('@')[-1]:
        return jsonify({"error": "Почта указана некорректно"}), 400

    password_hash = generate_password_hash(password, method='sha256')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s;", (username,))
        user = cursor.fetchone()
        if user:
            return jsonify({"error": "Пользователь уже существует"}), 409

        cursor.execute(
            "INSERT INTO users(username, email, password, created_at, graph_type_id) VALUES(%s, %s, %s, %s, %s)",
            (username, email, password_hash, datetime.datetime.now(), 1)
        )

        conn.commit()
        return jsonify({"message": "Пользователь успешно зарегистрирован"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

def token_required(f):
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]

        if not token:
            return jsonify({"error": "Токен отсутствует"}), 401

        try:
            data = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
            current_user = data["username"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен просрочен"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Неверный токен"}), 401

        return f(current_user, *args, **kwargs)

    return decorated