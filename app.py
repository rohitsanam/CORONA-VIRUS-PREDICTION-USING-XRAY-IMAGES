from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
#from gevent.pywsgi import WSGIServer

import sys
import os
import glob
global model,graph
import tensorflow as tf

app = Flask(__name__)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    New_pred = np.argmax(classes, axis=1)

    
    return New_pred


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)

        
        model = tf.keras.models.load_model('Covid_model.h5')

        preds = model_predict(file_path, model)
        
        if preds==[1]:
            p='Normal'
        else:
            p='Corona'
        
        return render_template('index.html', prediction_text='Patient condition: {}'.format(p))


if __name__ == '__main__':
     app.run(debug=True)