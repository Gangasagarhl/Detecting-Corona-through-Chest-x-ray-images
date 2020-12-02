from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('static\covid19_project.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1 # Used for removing cache

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image'] #imge variable recieves posted image from index

    img.save('static/{}.jpg'.format(COUNT))    # saves the image in static folder
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT)) # reading from static folder
    img_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)    
    img_arr = cv2.resize(img_arr,(200,200)) 

    
    print("")
    #img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 200,200,1) # reshaping model,to the required format
    prediction_class = model.predict_classes(img_arr) # predicting class of the prediction
    

    COUNT += 1
    return render_template('prediction.html', data=prediction_class)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



