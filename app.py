import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

#import ridge regreoson model and standard scaler pickle
ridge_model=pickle.load(open('models/nefty_ridge .pkl','rb'))
standard_scaler=pickle.load(open('models/nefty_scaler .pkl','rb'))

#route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def Predict_datapoint():
    if request.method=='POST':
        Open=float(request.form.get('Open'))
        Year=float(request.form.get('Year'))
        Month=float(request.form.get('Month'))
        Day=float(request.form.get('Day'))

        new_data_scaled=standard_scaler.transform([[Open,Year,Month,Day]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('Home.html',result=result[0])
            
    else:
        return render_template('Home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
