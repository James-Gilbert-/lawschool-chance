import sys
import os
import time
import traceback

from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
from collections import defaultdict
import pandas as pd
import numpy as np


app = Flask(__name__)


model_directory = 'models'

scaler = 'LSAT_GPA_SCALE'

feature_cols = "model_feats"

t14 = ["gulc","columbia","nyu","uva","yale","stanford","duke","cornell","northwestern","chicago","harvard","berkeley","michigan","penn"]

school_clf = defaultdict()
for school_name in t14:
    school_clf[school_name] = joblib.load(model_directory+"/"+school_name+"_model.joblib")

scaler = joblib.load(model_directory+"/"+scaler)

model_columns = joblib.load(model_directory+"/"+feature_cols)

@app.route('/')
def index():
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    if school_clf:
        try:
            scores=defaultdict()	
            json_data = request.get_json()
            query = pd.read_json(json_data)
            #create a vector of 0s to populate with filled-in data
            base = pd.DataFrame(0,index=np.arange(1),columns =model_columns)
            base.update(query)
            base[["lsat","gpa"]] = scaler.transform(base[["lsat","gpa"]].as_matrix())
            #populate the predictions
            for school_name in t14:
                scores[school_name +" chance of admission"]= str(school_clf[school_name].predict_proba(base)[:,1][0])
            prediction = scores

            return jsonify(prediction)

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        return("no model here")

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
