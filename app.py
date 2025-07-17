from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from PIL import Image
import numpy as np
import io
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# Monkeypox model info
MODEL_NAME = 'ResNet50V2'
HF_REPO = 'prrriiitam/ResNet50V2-01-monkeypox'
HF_FILENAME = 'ResNet50V2-01.keras'
INPUT_SIZE = (256, 256)
PREPROCESS_FUNC = resnet_preprocess
CLASS_NAMES = ['MonkeyPox', 'Others']

# Download and cache model once at startup
print("Downloading and loading model from Hugging Face...")
try:
    hf_token = os.getenv("HF_TOKEN")  # You should store this token as an environment variable on Render
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        token=hf_token
    )
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize(INPUT_SIZE)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = PREPROCESS_FUNC(image_array)

        # Predict
        prediction = model.predict(image_array)[0][0]
        predicted_class = CLASS_NAMES[1] if prediction > 0.5 else CLASS_NAMES[0]
        confidence = prediction if predicted_class == CLASS_NAMES[1] else 1 - prediction

        return jsonify({
            'prediction': predicted_class,
            'confidence': float(round(confidence * 100, 2))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))


