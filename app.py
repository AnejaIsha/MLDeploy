
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            LSTAT=float(request.form['LSTAT'])
            INDUS = float(request.form['INDUS'])
            NOX = float(request.form['NOX'])
            PTRATIO = float(request.form['PTRATIO'])
            RM = float(request.form['RM'])
            TAX = float(request.form['TAX'])
            DIS = float(request.form['DIS'])
            AGE = float(request.form['AGE'])
            filename = 'finalized_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            price=loaded_model.predict([[LSTAT,INDUS,NOX,PTRATIO,RM,TAX,DIS,AGE]])
            #price = loaded_model.predict([[2, 4, 5, 6, 7, 8, 9, 10]])
            print('prediction is', price)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=price)
        except Exception as e:
            print('The Exception message is: ',e)
            return e
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8003, debug=True)

	#app.run(debug=True) # running the app