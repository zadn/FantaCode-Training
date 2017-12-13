
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

model_file_name = 'model/model.pkl'
model_columns_file_name = 'model/model_columns.pkl'

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)

            for col in model_columns:
                if col not in query.columns:
                    query[col]=0.0
            
            
            
            prediction = list(clf.predict(query))

            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return('no model here')



if __name__ == '__main__':
   
    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train First')
        print(str(e))
        clf = None

    
    app.run(host='127.0.0.1', port=5000, debug=True)