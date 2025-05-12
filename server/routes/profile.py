from flask import Blueprint, jsonify, request
from functools import wraps
import jwt
from server.config import Config

profile_blueprint = Blueprint("profile", __name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]

        if not token:
            return jsonify({"message": "Токен отсутствует"}), 401

        try:
            data = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
            current_user = data.get("user")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Токен просрочен"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Неверный токен"}), 401

        return f(current_user, *args, **kwargs)

    return decorated

@profile_blueprint.route("/profile", methods=["GET"])
@token_required
def user_profile(current_user):
    return jsonify({"message": f"Добро пожаловать, {current_user}"}), 200