import sys
import os
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='food.h5'

model = load_model(MODEL_PATH)
    


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x=x/255
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        prediction=np.argmax(preds, axis=1)
        if prediction==0:
            prediction="Butter Chicken"
        elif prediction==1:
            prediction="Butter Naan"
        elif prediction==2:
            prediction="Chicken Fried Rice"
        elif prediction==3:
            prediction="Chole Bhature"

        elif prediction==4:
            prediction="Dahi Bhalla"
        elif prediction==5:
            prediction="Dal Malhani"
        elif prediction==6:
            prediction="Idli"
        elif prediction==7:
            prediction="Kachori"
        elif prediction==8:
            prediction="Rasgulla"
        else:
            prediction="Samosa"

            
        
        return render_template('index.html',result = prediction)

if __name__ == '__main__':
    app.run(debug=True)