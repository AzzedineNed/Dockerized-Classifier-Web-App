import os
import numpy as np
import random
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Set up the Flask application
app = Flask(__name__)

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)

# Load ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image_class(image_path, threshold=0.5):
    img = load_and_preprocess_image(image_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[class_idx]
    confidence = preds[0][class_idx]
    if confidence < threshold:
        return "Neither Cat nor Dog", confidence
    return predicted_class, confidence

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    predicted_class, confidence = predict_image_class(file_path)

    return render_template('result.html', predicted_class=predicted_class, confidence=confidence, image_path=file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
