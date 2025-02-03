# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
import os

# Load and preprocess the MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Reshape data to add a channel dimension (required for CNN)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, y_train, x_test, y_test

# Build the CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# Save the model
def save_model(model, filename):
    model.save(filename)

# Load the model
def load_model(filename):
    return tf.keras.models.load_model(filename)

# Preprocess an input image for prediction
def preprocess_image(image):
    # Resize the image to 28x28 and convert to grayscale
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the image (MNIST digits are white on black background)
    image = cv2.bitwise_not(image)
    
    # Reshape and normalize
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    return image

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    
    return jsonify({'predicted_class': int(predicted_class)})

# Main function
if __name__ == '__main__':
    # Load data
    x_train, y_train, x_test, y_test = load_data()
    
    # Build and train the model
    model = build_model()
    train_model(model, x_train, y_train, x_test, y_test, epochs=10)
    
    # Save the model
    save_model(model, 'mnist_cnn_model.h5')
    
    # Load the model (optional, if you want to load a pre-trained model)
    # model = load_model('mnist_cnn_model.h5')
    
    # Run the Flask app
    app.run(debug=True)
    