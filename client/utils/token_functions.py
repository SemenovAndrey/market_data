import sys
import json
import os
from flask import request, jsonify
import jwt

from server.config import Config

TOKEN_FILE = "client/token.json"

def save_token(token):
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as token_file:
        json.dump({"token": token}, token_file)
    print("Токен сохранен: ", token)

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as token_file:
            data = json.load(token_file)
            return data.get("token")
    else:
        save_token("")
        return None

def get_token():
    try:
        with open(TOKEN_FILE, "r") as token_file:
            data = json.load(token_file)
            return data.get("token")
    except FileNotFoundError:
        return None

# def token_required(f):
#     from functools import wraps
#
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = None
#
#         if "Authorization" in request.headers:
#             token = request.headers["Authorization"].split(" ")[1]
#
#         if not token:
#             return jsonify({"error": "Токен отсутствует"}), 401
#
#         try:
#             data = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
#             current_user = data["username"]
#         except jwt.ExpiredSignatureError:
#             return jsonify({"error": "Токен просрочен"}), 401
#         except jwt.InvalidTokenError:
#             return jsonify({"error": "Неверный токен"}), 401
#
#         return f(current_user, *args, **kwargs)
#
#     return decorated