import requests
from flask import Blueprint, jsonify, request
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import math

from server.db import get_db_connection
from server.routes.auth import token_required

# MOEX_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"

assets_blueprint = Blueprint('assets', __name__)

@assets_blueprint.route("/list", methods=['GET'])
# @token_required
# def assets_list(current_user):
def assets_list():
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
# @token_required
# def latest_price(current_user, symbol):
def latest_price(symbol):
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
# def get_asset_history(current_user, symbol):
def get_asset_history(symbol):
    interval = request.args.get("interval", "1d")
    start_str = request.args.get("start")
    end_str = request.args.get("end")

    today = datetime.today()

    ticker = yf.Ticker(symbol)
    ticker_found = True
    data_frame = None

    try:
        data_frame = ticker.history(period="max")
    except Exception as e:
        ticker_found = False
        print("Тикер не найден: " + str(e))

    if ticker_found and not data_frame.empty:
        last_row = data_frame.iloc[-1]

        check_moex = False
        if (math.isnan(last_row["Open"]) and math.isnan(last_row["High"]) and math.isnan(last_row["Low"])
                and math.isnan(last_row["Close"]) and last_row["Volume"] == 0):
            check_moex = True

        if not check_moex:
            first_date = data_frame.index[0].to_pydatetime()

            if start_str and end_str:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
            else:
                start_date = first_date
                end_date = today

            if start_date.date() != first_date.date() or end_date.date() != today.date():
                if start_date.date() < first_date.date():
                    return jsonify({
                        "error":
                            f"Дата начала торгов для {symbol} — {first_date.date()}. Вы выбрали слишком раннюю дату."
                    }), 400

                if start_date.date() > end_date.date():
                    return jsonify({
                        "error": f"Начальная дата не может быть позже конечной даты."
                    }), 400

                if end_date.date() > today.date():
                    return jsonify({
                        "error": f"Конечная дата не может быть позже сегодняшнего дня {today.date()}."
                    }), 400

                data_frame = ticker.history(interval=interval, start=start_date.date(), end=end_date.date())

            if not data_frame.empty:
                data = []
                for index, row in data_frame.iterrows():
                    data.append({
                        "timestamp": index.isoformat(),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"])
                    })

                return jsonify({
                    "symbol": symbol,
                    "interval": interval,
                    "source": "Yahoo Finance",
                    "data": data
                }), 200

    moex_symbol = convert_to_moex_symbol(symbol)
    moex_url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{moex_symbol}.json"

    moex_response = requests.get(moex_url, timeout=5)
    if moex_response.status_code != 200:
        return jsonify({"error": "MOEX API недоступен"}), 502

    moex_data = moex_response.json()
    rows = moex_data.get("history", {}).get("data", [])
    columns = moex_data.get("history", {}).get("columns", [])

    if not rows or not columns:
        print("Ошибка здесь")
        return jsonify({"error": f"Нет данных по тикеру {symbol}."}), 404

    col_map = {name: idx for idx, name in enumerate(columns)}

    dates = [datetime.strptime(row[col_map["TRADEDATE"]], "%Y-%m-%d") for row in rows]
    first_moex_date = min(dates)

    if start_str and end_str:
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
    else:
        start_date = first_moex_date
        end_date = today

    if start_date.date() != first_moex_date.date() or end_date.date() != today.date():
        if start_date.date() < first_moex_date.date():
            return jsonify({
                "error":
                f"Дата начала торгов для {symbol} на MOEX — {first_moex_date.date()}. Вы выбрали слишком раннюю дату."
            }), 400

        if start_date.date() > end_date.date():
            return jsonify({
                "error": f"Начальная дата не может быть позже конечной даты."
            }), 400

        if end_date.date() > today.date():
            return jsonify({
                "error": f"Конечная дата не может быть позже сегодняшнего дня {today.date()}."
            }), 400

        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": 24
        }

        moex_filtered_response = requests.get(moex_url, params=params, timeout=5)
        if moex_filtered_response.status_code != 200:
            return jsonify({"error": "MOEX API недоступен (фильтр)"}), 502

        moex_data = moex_filtered_response.json()

        rows = moex_data.get("history", {}).get("data", [])
        if not rows:
            print("Ошибка здесь 1")
            return jsonify({"error": "Нет данных за выбранный период"}), 404

    data = []
    for row in rows:
        data.append({
            "timestamp": row[col_map["TRADEDATE"]],
            "open": float(row[col_map["OPEN"]]),
            "high": float(row[col_map["HIGH"]]),
            "low": float(row[col_map["LOW"]]),
            "close": float(row[col_map["CLOSE"]]),
            "volume": float(row[col_map["VOLUME"]]),
        })

    return jsonify({
        "symbol": moex_symbol,
        "interval": "1d",
        "source": "MOEX",
        "data": data
    }), 200

@assets_blueprint.route("/<symbol>/indicators", methods=["GET"])
# @token_required
# def get_indicators(current_user, symbol):
def get_indicators(symbol):
    try:
        interval = request.args.get("interval", "1d")
        start_str = request.args.get("start")
        end_str = request.args.get("end")
        requested_param = request.args.get("indicators")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)

        if start_str:
            start_date = datetime.strptime(start_str, "%Y-%m-%d")

        if end_str:
            end_date = datetime.strptime(end_str, "%Y-%m-%d")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM indicators_names")
        available = [row[0].strip().upper() for row in cursor.fetchall()]

        if requested_param:
            requested = [ind.strip().upper() for ind in requested_param.upper().split(",") if ind.strip().upper() in available]
        else:
            return jsonify({"error": "Не выбран индикатор"})

        ticker = yf.Ticker(symbol)
        data_frame = ticker.history(interval=interval, start=start_date, end=end_date)

        if data_frame.empty:
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

            data = response.json()
            rows = data.get("history", {}).get("data", [])
            columns = data.get("history", {}).get("columns", [])

            if not rows or not columns:
                return jsonify({"error": f"Ничего не найдено для {symbol}"}), 404

            col_map = {col: i for i, col in enumerate(columns)}
            records = []
            for row in rows:
                records.append({
                    "timestamp": pd.to_datetime(row[col_map["TRADEDATE"]]),
                    "open": row[col_map["OPEN"]],
                    "high": row[col_map["HIGH"]],
                    "low": row[col_map["LOW"]],
                    "close": row[col_map["CLOSE"]],
                    "volume": row[col_map["VOLUME"]],
                })
            data_frame = pd.DataFrame(records).set_index("timestamp")

        if data_frame.empty:
            return jsonify({"error": f"Нет данных для актива {symbol}"}), 404

        data_frame = data_frame.rename(columns=str.lower)
        data_frame["timestamp"] = data_frame.index

        result = {}

        if "SMA" in requested:
            data_frame["sma_14"] = data_frame["close"].rolling(window=14).mean()
            result["SMA"] = data_frame[["timestamp", "sma_14"]].dropna().tail(30).round(2).to_dict(orient="records")

        if "EMA" in requested:
            data_frame["ema_14"] = data_frame["close"].ewm(span=14, adjust=False).mean()
            result["EMA"] = data_frame[["timestamp", "ema_14"]].dropna().tail(30).round(2).to_dict(orient="records")

        if "RSI" in requested:
            delta = data_frame["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            data_frame["rsi_14"] = 100 - (100 / (1 + rs))
            result["RSI"] = data_frame[["timestamp", "rsi_14"]].dropna().tail(30).round(2).to_dict(orient="records")

        if "MACD" in requested:
            ema12 = data_frame["close"].ewm(span=12, adjust=False).mean()
            ema26 = data_frame["close"].ewm(span=26, adjust=False).mean()
            data_frame["macd"] = ema12 - ema26
            data_frame["macd_signal"] = data_frame["macd"].ewm(span=9, adjust=False).mean()
            result["MACD"] = data_frame[["timestamp", "macd", "macd_signal"]].dropna().tail(30).round(2).to_dict(
                orient="records")

        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "indicators": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

def convert_to_moex_symbol(symbol):
    return symbol.replace(".ME", "")