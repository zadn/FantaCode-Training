
from flask import Flask, request, jsonify
import pandas as pd
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
            prediction_values = []

            for items in prediction:
                prediction_values.append(int(items))

            print(prediction)
            return jsonify({'prediction': list(prediction_values)})

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

    else:
        print('train first')
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    digits = load_iris()
    data = digits.data
    targ = digits.target

    X_train, X_test, y_train, y_test = train_test_split(data, targ, test_size=0.1, random_state=0)

    model = LogisticRegression()
    model = model.fit(X_train, y_train)

    joblib.dump(model, model_file_name)

    return 'Trained Successfully'


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