from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image

# === Setup ===
app = Flask(__name__)
CORS(app)

# === Load Model ===
MODEL_PATH = os.path.join('model', 'xception_plant_model.h5')
model = load_model(MODEL_PATH)

# === Load Class Names ===
CLASS_NAMES_PATH = os.path.join('model', 'class_names.txt')
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# === Upload folder ===
UPLOAD_FOLDER = os.path.join('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Serve the HTML page ===
@app.route('/')
def index():
    return render_template('index.html')

# === Predict route ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)
