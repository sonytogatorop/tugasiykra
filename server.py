from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
app = Flask(__name__, template_folder='templates')

@app.route('/')
def student():
   return render_template("home.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    result = float(ValuePredictor(to_predict_list))
    return render_template("home.html",result = result)

if __name__ == '__main__':
    app.run(debug = True)
