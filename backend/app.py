from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image

# === App Setup ===
app = Flask(__name__)
CORS(app)

# === Load Model ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'xception_plant_model.h5')
model = load_model(MODEL_PATH)

# === Load Class Names ===
CLASS_NAMES_PATH = os.path.join(BASE_DIR, '..', 'model', 'class_names.txt')

def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []

class_names = load_class_names()

# === Upload Folder ===
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Prediction Endpoint ===
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

        # Preprocess the image
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
