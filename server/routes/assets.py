import requests
from flask import Blueprint, jsonify, request
import yfinance as yf
from datetime import datetime, timedelta

from server.db import get_db_connection
from server.routes.auth import token_required

# MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"

assets_blueprint = Blueprint('assets', __name__)

@assets_blueprint.route("/list", methods=['GET'])
@token_required
def assets_list(current_user):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, name FROM assets;")
        assets = cursor.fetchall()
        conn.close()
        return jsonify([{"symbol": asset[0], "name": asset[1]} for asset in assets])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assets_blueprint.route("/<symbol>/latest", methods=['GET'])
@token_required
def latest_price(current_user, symbol):
    symbol = symbol.upper()
    try:
        asset = yf.Ticker(symbol)
        data = asset.history(period="1d")
        # print(data)
        if not data.empty:
            latest = data.iloc[-1]
            return jsonify({
                "symbol": symbol,
                "open": latest["Open"],
                "close": latest["Close"],
                "high": latest["High"],
                "low": latest["Low"],
                "volume": latest["Volume"]
            })

        moex_symbol = convert_to_moex_symbol(symbol)
        moex_url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{moex_symbol}.json"

        response = requests.get(moex_url)
        if response.status_code == 200:
            data = response.json()
            market_data = data.get("marketdata",{}).get("data", [])
            columns = data.get("marketdata", {}).get("columns", [])

            if market_data and columns:
                column_map = {name: idx for idx, name in enumerate(columns)}
                latest_data = market_data[1]
                # return jsonify({"columns": columns, "raw_data": market_data})

                return jsonify({
                    "symbol": moex_symbol,
                    "source": "MOEX",
                    "open": latest_data[column_map["OPEN"]],
                    "close": latest_data[column_map["LAST"]],
                    "high": latest_data[column_map["HIGH"]],
                    "low": latest_data[column_map["LOW"]],
                    "volume": latest_data[column_map["VALUE"]]
                })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@assets_blueprint.route("/<symbol>/history", methods=["GET"])
# @token_required
def get_asset_history(symbol):
    try:
        interval = request.args.get("interval", "1d")
        start_str = request.args.get("start")
        end_str = request.args.get("end")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=30)

        if start_str:
            start_date = datetime.strptime(start_str, "%Y-%m-%d")

        if end_str:
            end_date = datetime.strptime(end_str, "%Y-%m-%d")

        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, start=start_date, end=end_date)

        if not df.empty:
            data = []
            for index, row in df.iterrows():
                data.append({
                    "timestamp": index.isoformat(),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"]
                })

            return jsonify({
                "symbol": symbol,
                "interval": interval,
                "data": data
            }), 200

        moex_symbol = convert_to_moex_symbol(symbol)
        moex_url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{moex_symbol}.json"

        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": 24
        }

        response = requests.get(moex_url, params=params, timeout=5)
        if response.status_code != 200:
            return jsonify({"error": "MOEX API недоступен"}), 502

        moex_data = response.json()
        rows = moex_data.get("history", {}).get("data", [])
        columns = moex_data.get("history", {}).get("columns", [])

        if not rows or not columns:
            return jsonify({"error": f"Нет исторических данных на MOEX для {symbol}"}), 404

        col_map = {name: idx for idx, name in enumerate(columns)}
        result = []
        for row in rows:
            result.append({
                "timestamp": row[col_map["TRADEDATE"]],
                "open": row[col_map["OPEN"]],
                "high": row[col_map["HIGH"]],
                "low": row[col_map["LOW"]],
                "close": row[col_map["CLOSE"]],
                "volume": row[col_map["VOLUME"]],
            })

        return jsonify({
            "symbol": moex_symbol,
            "interval": "1d",
            "source": "MOEX",
            "data": result
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def convert_to_moex_symbol(symbol):
    return symbol.replace(".ME", "")