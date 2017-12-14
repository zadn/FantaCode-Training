
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.externals import joblib
import traceback

app = Flask(__name__)

model_file_name = 'model/model_iris.pkl'




@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)

            prediction = (clf.predict(query))

            print(prediction)
            return jsonify({'prediction': list(prediction)})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

    else:
        print('train first')
        return 'no model here'


if __name__ == '__main__':

    clf = None

    try:
        clf = joblib.load(model_file_name)
        print('Model Loaded')

    except Exception as e:
        print('No model here')
        print('Train First')
        print(str(e))
        clf = None

    app.run(host='127.0.0.1', port=3000, debug=True)