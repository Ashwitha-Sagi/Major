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

# === Load Model and Class Names ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'xception_plant_model.h5')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, '..', 'model', 'class_names.txt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    raise RuntimeError(f"Failed to load class names: {e}")

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

        # Preprocess image
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        pred_index = int(np.argmax(preds))
        pred_class = class_names[pred_index]
        return jsonify({'prediction': pred_class})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# === Main ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render uses $PORT
    app.run(debug=False, host='0.0.0.0', port=port)
