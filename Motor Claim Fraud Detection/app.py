import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from get_values_from_config_file import port
from single_record_prediction import predict_single_json_record

app = Flask(__name__)
CORS(app, support_credentials=True)
# app.secret_key = ""
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # 2 space indentation
app.config['JSON_SORT_KEYS'] = False  # avoids jsonify to sort the keys in alphabetical manner


@app.route('/')
@cross_origin(supports_credentials=True)
def home_page():
    result = [
        {
            'Created By': 'Azentio Software Pvt Ltd : https://www.azentio.com/',
            'description': 'A Motor Claim Fraud detection model using a kaggle dataset',
        }
    ]
    return jsonify(result)


# predict will provide the response for the function created in Model Prediction.py
@app.route('/predict', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def predict_page():
    if request.method == 'POST':
        try:
            data = request.json
            if data is not None:
                response = jsonify(predict_single_json_record(data))
                return response
            else:
                return "Data is not defined."
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return f"There is no model available to use.Please check your model file."


if __name__ == '__main__':
    import socket

    hostname = socket.gethostname()
    port = port
    app.run(host=hostname, port=port, debug=False)
