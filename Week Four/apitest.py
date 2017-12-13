
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib
import traceback
import pickle


app = Flask(__name__)

#model_file_name = 'model/model.pkl'
model_columns_file_name = 'model/model_columns.pkl'


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)

            for col in model_columns:
                if col not in query.columns:
                    query[col] = 0.0

            prediction = list(clf.predict(query))

            for col in query.columns:
                print(query[col])
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'


#@app.route('/train', methods=['GET'])

if __name__ == '__main__':

    import numpy as np
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from patsy import dmatrices

    data = sm.datasets.fair.load_pandas().data
    data['affair'] = (data.affairs > 0).astype(int)
    y, X = dmatrices(
        'affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',
        data, return_type='dataframe')
    X = X.rename(columns={'C(occupation)[T.2.0]': 'occ_2',
                          'C(occupation)[T.3.0]': 'occ_3',
                          'C(occupation)[T.4.0]': 'occ_4',
                          'C(occupation)[T.5.0]': 'occ_5',
                          'C(occupation)[T.6.0]': 'occ_6',
                          'C(occupation_husb)[T.2.0]': 'occ_husb_2',
                          'C(occupation_husb)[T.3.0]': 'occ_husb_3',
                          'C(occupation_husb)[T.4.0]': 'occ_husb_4',
                          'C(occupation_husb)[T.5.0]': 'occ_husb_5',
                          'C(occupation_husb)[T.6.0]': 'occ_husb_6', })
    y = np.ravel(y)
    model = LogisticRegression()
    model = model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    saved = pickle.dumps(model2)

    print('Successfully Trained')


    try:

        clf = pickle.loads(saved)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train First')
        print(str(e))
        clf = None

    app.run(host='127.0.0.1', port=5000, debug=True)