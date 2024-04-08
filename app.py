from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define a Flask app
app = Flask(__name__)

# Path to your trained model
MODEL_PATH = 'discriminator_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define your emotion classes
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32), color_mode="grayscale")
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        try:
            preds = model_predict(file_path, model)
            pred_index = np.argmax(preds[0])  # Assuming model outputs one prediction per image
            result = EMOTIONS[pred_index]  # Map the prediction to the corresponding emotion
        except Exception as e:
            result = str(e)  # Display the error as the result
        
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
