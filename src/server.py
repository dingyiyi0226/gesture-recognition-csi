from http import HTTPStatus
import time

from flask import Flask
from flask import request
from flask import make_response

from inference_server import InferenceServer

def create_app():
    app = Flask(__name__)
    app.secret_key = 'secret_key'

    fileroot = 'data_realtime/'
    server = InferenceServer(root=fileroot)

    @app.route('/')
    def home():
        return make_response('Hello', HTTPStatus.OK)

    @app.route('/gesture', methods=["POST"])
    def gesture():
        file = request.files['csi']

        filename = f'{int(time.time())}.pcap'
        file.save(fileroot+filename)

        pred_all = server.inference(files=filename, verbose=False)
        return make_response(pred_all[0], HTTPStatus.OK)

    return app
