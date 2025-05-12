from flask import Blueprint, jsonify

from server.db import get_db_connection

db_requests_blueprint = Blueprint('db_queries', __name__)

@db_requests_blueprint.route("/forecasting_methods", methods=["GET"])
# @token_required
# def get_forecasting_methods(current_user):
def get_forecasting_methods():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM forecasting_methods WHERE is_actively = true;")
        rows = cursor.fetchall()
        methods = [row[0] for row in rows]
        cursor.close()
        conn.close()
        return jsonify({"methods": methods}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@db_requests_blueprint.route("/indicators_names", methods=["GET"])
# @token_required
# def get_indicators_names(current_user):
def get_indicators_names():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM indicators_names WHERE is_actively = true;")
        rows = cursor.fetchall()
        indicators = [row[0] for row in rows]
        cursor.close()
        conn.close()
        return jsonify({"indicators": indicators}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
