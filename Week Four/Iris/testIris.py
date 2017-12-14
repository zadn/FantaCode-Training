import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_iris()
data = digits.data
targ = digits.target
X_train, X_test, y_train, y_test = train_test_split(data, targ, test_size=0.1)


model = LogisticRegression()
model = model.fit(X_train, y_train)

model_file_name = 'model/model_iris.pkl'

joblib.dump(model, model_file_name)

