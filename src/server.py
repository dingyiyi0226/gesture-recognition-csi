from http import HTTPStatus
import time

from flask import Flask
from flask import request
from flask import make_response

from inference import inference

def create_app():
    app = Flask(__name__)
    app.secret_key = 'secret_key'

    @app.route('/')
    def home():
        return make_response('Hello', HTTPStatus.OK)

    @app.route('/gesture', methods=["POST"])
    def gesture():
        file = request.files['csi']

        fileroot = 'data_realtime/'
        filename = f'{int(time.time())}.pcap'

        file.save(fileroot+filename)
        pred_all = inference(root=fileroot, files=filename, verbose=False)
        return make_response(pred_all[0], HTTPStatus.OK)

    return app
