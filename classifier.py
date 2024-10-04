import tensorflow as tf
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load ResNet50 without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with 1024 neurons and ReLU activation
x = Dense(1024, activation='relu')(x)

# Add a logistic layer with 2 classes (binary classification for cat and dog)
predictions = Dense(2, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base ResNet50 model so only the new layers are trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess an image using OpenCV (cv2)
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Make predictions with the model
def predict_image_class(image_path, threshold=0.6):
    img = load_and_preprocess_image(image_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[class_idx]
    confidence = preds[0][class_idx]
    if confidence < threshold:
        return "Neither Cat nor Dog", confidence
    return predicted_class, confidence

# Visualize the image and prediction
def display_image_and_prediction(image_path, threshold=0.6):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predicted_class, confidence = predict_image_class(image_path, threshold)
    plt.imshow(img_rgb)
    plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()

# Example: Use the classifier to predict a class for a new image
image_path = 'static\Dog.jpg'  # Change to your image path
display_image_and_prediction(image_path, threshold=0.5)