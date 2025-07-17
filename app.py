from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
from huggingface_hub import hf_hub_download


# Preprocessing functions for different models
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.vgg16 import preprocess_input as VGG16_preprocess


app = Flask(__name__)
CORS(app)

# Define available models with their details
MODEL_CONFIG = {
    'ResNet50V2': {
        'hf_repo': 'prrriiitam/ResNet50V2-01-monkeypox',
        'hf_filename': 'ResNet50V2-01.keras',
        'input_size': (256, 256),
        'preprocess': resnet_preprocess
    }

}

# Class names

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'modelName' not in request.form:
        return jsonify({'error': 'Missing image or model_name'}), 400

    file = request.files['file']
    model_name = request.form['modelName']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if model_name not in MODEL_CONFIG:
        return jsonify({'error': f'Model {model_name} not found'}), 400
    
    if model_name in ['VGG16']: 
        try: 
            class_labels = [ 'Chickenpox', 'Cowpox', 'HFMD', 'Healthy','Measles', 'Monkeypox']
            config = MODEL_CONFIG[model_name]
            model_path = config['path']
            input_size = config['input_size']
            preprocess_func = config['preprocess']
            model = load_model(model_path)
        # Process image
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image = image.resize(input_size)
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_func(image_array)
            preds = model.predict(image_array)
            class_index = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))
            return jsonify({
                'prediction': class_labels[class_index],
                'confidence': round(confidence * 100, 2)  # Convert to percentage
            }) 
        except Exception as e: 
            return jsonify({'error': str(e)}), 500


    else: 
        try: 
           class_names = ['MonkeyPox', 'Others']
           config = MODEL_CONFIG[model_name]
           
           
           if 'hf_repo' in config: 
               model_path = hf_hub_download(
               repo_id=config['hf_repo'],
               filename=config['hf_filename'],
               token=os.getenv("HF_TOKEN")  ) # Or directly set your token here if needed

           else:
               model_path = config['path']

           input_size = config['input_size']
           preprocess_func = config['preprocess']
           # Load model (optional: cache if you want)
           model = load_model(model_path)
           # Process image
           image = Image.open(io.BytesIO(file.read())).convert('RGB')
           image = image.resize(input_size)
           image_array = img_to_array(image)
           image_array = np.expand_dims(image_array, axis=0)
           image_array = preprocess_func(image_array)
           prediction = model.predict(image_array)[0][0]

           predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
           confidence = prediction if predicted_class == class_names[1] else 1 - prediction

           return jsonify({
            'prediction': predicted_class,
            'confidence': float(round(confidence * 100, 2))
           })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("facing an error")
            return jsonify({'facing an error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)









# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from PIL import Image
# import numpy as np
# import io
# import os
# import gdown

# # Preprocessing functions for different models
# from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
# from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
# from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
# from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
# from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# app = Flask(__name__)
# CORS(app)

# MODEL_CONFIG = {
#     'ResNet50V2-01.keras': {
#         'drive_id': '1MM5MrbVenlwj2NjbQHgpgwgyedwrSNe2',  
#         'path': 'models/ResNet50V2-01.keras',
#         'input_size': (256, 256),
#         'preprocess': resnet_preprocess
#     },
#     'mdl_wts.keras': {
#         'drive_id': '1KEWNd05RKuebJjPxwZqdE-iiSURg975g',
#         'path': 'models/mdl_wts.keras',
#         'input_size': (224, 224),
#         'preprocess': mobilenet_preprocess
#     },
#     'densenet201_mpox_best.keras': {
#         'drive_id': '1eS90FBBFnh8VyMqUTqBz6clzG6pMegRg',
#         'path': 'models/densenet201_mpox_best.keras',
#         'input_size': (224, 224),
#         'preprocess': densenet_preprocess
#     },
#     'EfficientnetB3weights.keras': {
#         'drive_id': '1l6cktrCphQ2rAdb1tgSPKFAL4jEaANSs',
#         'path': 'models/EfficientnetB3weights.keras',
#         'input_size': (224, 224),
#         'preprocess': efficientnet_preprocess
#     },
#     'ConvNeXtmodel.keras': {
#         'drive_id': '1Yl33_6vc9NaTJTvfYfxeiAW1ilEqoMo_',
#         'path': 'models/ConvNeXtmodel.keras',
#         'input_size': (224, 224),
#         'preprocess': convnext_preprocess
#     }
# }

# class_names = ['MonkeyPox', 'Others']

# def download_model_if_needed(model_path, drive_id):
#     if not os.path.exists(model_path):
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         gdown.download(f"https://drive.google.com/uc?id={drive_id}", model_path, quiet=False)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files or 'model_name' not in request.form:
#         return jsonify({'error': 'Missing image or model_name'}), 400

#     file = request.files['image']
#     model_name = request.form['model_name']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if model_name not in MODEL_CONFIG:
#         return jsonify({'error': f'Model {model_name} not found'}), 400

#     try:
#         config = MODEL_CONFIG[model_name]
#         model_path = config['path']
#         input_size = config['input_size']
#         preprocess_func = config['preprocess']
#         drive_id = config['drive_id']

#         download_model_if_needed(model_path, drive_id)

#         model = load_model(model_path)

#         image = Image.open(io.BytesIO(file.read())).convert('RGB')
#         image = image.resize(input_size)
#         image_array = img_to_array(image)
#         image_array = np.expand_dims(image_array, axis=0)
#         image_array = preprocess_func(image_array)

#         prediction = model.predict(image_array)[0][0]
#         predicted_class = class_names[1] if prediction > 0.5 else class_names[0]

#         return jsonify({'prediction': predicted_class})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
