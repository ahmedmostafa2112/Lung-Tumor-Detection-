import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

@app.route('/', methods=["post"])
def index():
    # load the model
    model_path = "LCDCNN.h5"
    loaded_model = tf.keras.models.load_model(model_path)


    # loadind the photo
    image = request.files['image']
    image.save('img.jpg')


    image = cv2.imread('img.jpg')

    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((64, 64))
    expand_input = np.expand_dims(resize_image, axis=0)
    input_data = np.array(expand_input)
    input_data = input_data / 255

    pred = loaded_model.predict(input_data)

    results = ''

    if pred[0][0] > 0.5:
        results = "NO"
    else:
        results = "YES"


    return jsonify(results)
if __name__ == "__main__":
    app.run('0.0.0.0',9090)
