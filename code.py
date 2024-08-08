import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import zipfile
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to your zip file in Google Drive
zip_path = '/content/drive/MyDrive/food-101.zip'

# Specify the directory to extract the dataset
extract_dir = '/content/food-101'

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Now we can define our data loading function
def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(os.path.join(data_dir, 'images')))

    for class_name in class_names:
        class_dir = os.path.join(data_dir, 'images', class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(class_names.index(class_name))

    X = np.array(images)
    y = np.array(labels)

    # Normalize pixel values to be between 0 and 1
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, class_names

# The rest of the code remains largely the same

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))
    return history

# Main execution
if __name__ == "__main__":
    data_dir = extract_dir
    X_train, y_train, X_test, y_test, class_names = load_and_preprocess_data(data_dir)

    input_shape = X_train.shape[1:]  # (224, 224, 3)
    num_classes = len(class_names)  # Should be 101 for Food-101

    model = build_model(input_shape, num_classes)
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Save the model
    model.save("/content/drive/MyDrive/food_recognition_model.h5")

# Function to predict food from an image
def predict_food(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_food = class_names[predicted_class]

    return predicted_food

# Placeholder function for calorie estimation
def estimate_calories(food_item):
    # This is a placeholder. You'd need to implement actual calorie estimation.
    calorie_dict = {
        "apple pie": 237,
        "pizza": 266,
        # ... add more food items and their calorie content
    }
    return calorie_dict.get(food_item, "Calorie information not available")

# User interface
while True:
    image_path = input("Enter the path to your food image (or 'q' to quit): ")
    if image_path.lower() == 'q':
        break

    predicted_food = predict_food(image_path, model, class_names)
    calories = estimate_calories(predicted_food)

    print(f"Predicted food: {predicted_food}")
    print(f"Estimated calories: {calories}")