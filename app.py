from flask import Flask,request,render_template
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('Gradient_Boosting.pkl','rb'))
scaler=pickle.load(open('Scaler.pkl','rb'))

# Prediction function
def predict(model, scaler,model_year, milage, fuel_type, engine, transmission,ext_col, int_col, accident):
    
    features=np.array([[model_year, milage, fuel_type, engine, transmission,ext_col, int_col, accident]])
    
    scaled_features=scaler.transform(features)
    
    result=model.predict(scaled_features)
    
    return result[0]



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_route():
    if request.method=='POST':
        model_year=int(request.form['model_year'])
        milage=int(request.form['milage'])
        fuel_type=int(request.form['fuel_type'])
        engine=int(request.form['engine'])
        transmission=int(request.form['transmission'])
        ext_col=int(request.form['ext_col'])
        int_col=int(request.form['int_col'])
        accident=int(request.form['accident'])

        prediction=predict(model,scaler,model_year, milage, fuel_type, engine, transmission,ext_col, int_col, accident)
        
        return render_template('index.html',prediction=prediction)





if __name__=='__main__':
    app.run(debug=True)