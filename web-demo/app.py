from flask import Flask, request, render_template
from translate_client import NJUNMTClient

app = Flask(__name__)
# bootstrap = Bootstrap(app)
# runner = TranslationRunner()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server_ip", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default="1234")
args = parser.parse_args()

client = NJUNMTClient((args.server_ip, args.server_port))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def home_translation():
    if request.form["action"] == "translate":
        source = request.form['source']
        print(source)
        user_ip = request.remote_addr
        response = client.request("translate", source, user_ip, True)
        print(response)
        return render_template('index.html', source=source, translation=response['translation'],
                               model_dirs=response["model_info"]["model_dir"])
    elif request.form["action"] == "load_model":
        model_dir = request.form['model_dir']
        user_ip = request.remote_addr
        response = client.request("reload", model_dir, user_ip, True)
        print(response)
        return render_template('index.html', model_dirs=response["model_info"]["model_dir"])


if __name__ == '__main__':

    app.run(host="172.16.1.25", port=9999, debug=False, processes=8)
