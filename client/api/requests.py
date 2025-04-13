import requests

from client.utils.token_functions import get_token, TOKEN_FILE, load_token

SERVER_URL = "http://127.0.0.1:5000"

def login(username, password):
    url = f"{SERVER_URL}/auth/login"

    try:
        response = requests.post(url, json={"username": username, "password": password})
        if response.status_code == 200:
            return response.json().get("token")
        else:
            print("Ошибка: ", response.json().get("error"))
            return None
    except Exception as e:
        print("Ошибка подключения к серверу", e)
        return None

def register(username, email, password):
    url = f"{SERVER_URL}/auth/register"

    try:
        response = requests.post(url, json={"username": username, "email": email, "password": password})
        if response.status_code == 200:
            return response.json().get("token")
        else:
            print("Ошибка: ", response.json().get("error"))
            return None
    except Exception as e:
        print("Ошибка подключения к серверу", e)
        return None

def get_user_profile():
    url = f"{SERVER_URL}/user/profile"
    token = get_token()
    if not token:
        print("Токен отсутствует, необходимо войти")
        return None

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print("Ошибка: ", response.json().get("error"))
            return None
    except Exception as e:
        print("Ошибка подключения к серверу: ", e)
        return None

def get_asset_history(symbol):
    token = load_token()
    if not token:
        return None

    url = f"{SERVER_URL}/assets/{symbol}/history"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("data", [])
    except Exception as e:
        print("Ошибка запроса: ", e)

    return None
