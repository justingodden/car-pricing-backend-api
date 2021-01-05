import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict


app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST'])
def index():
    if request.is_json:
        req = request.get_json()
        price = str(predict(req))
        return price


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
