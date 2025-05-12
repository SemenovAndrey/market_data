from flask import Flask, jsonify

from server.config import Config
from server.routes.auth import auth_blueprint
from server.routes.assets import assets_blueprint
from server.routes.profile import profile_blueprint
from server.routes.db_requests import db_requests_blueprint

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = Config.SECRET_KEY

app.register_blueprint(auth_blueprint, url_prefix='/auth')
app.register_blueprint(assets_blueprint, url_prefix='/assets')
app.register_blueprint(profile_blueprint, url_prefix='/profile')
app.register_blueprint(db_requests_blueprint, url_prefix='/db_requests')

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Добро пожаловать"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
