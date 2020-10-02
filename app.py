from flask import Flask,request,render_template,url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model_lr.pkl','rb'))
@app.route('/')
def Welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    output = model.predict([np.array(values)])
    output = round(output[0][0],2)
    return render_template("index.html",percentage='Your percentage is {}%'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
