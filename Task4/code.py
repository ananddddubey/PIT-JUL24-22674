from google.colab import drive
import zipfile
import os
import random
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

drive.mount('/content/drive')

# Google Drive path
zip_path = '/content/drive/MyDrive/archive.zip'
# Directory to extract to
extract_path = '/content/Hand_gesture_dataset'

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Files extracted successfully.")
else:
    print("Files already extracted.")

images = []
labels = []

# Adjust the directory traversal
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            images.append(img_path)
            # Extract label from the parent directory name
            label = os.path.basename(os.path.dirname(img_path))
            labels.append(label)

print(images[:5])
print(labels[:5])

def plot_one_image_per_label(image_paths, labels, img_size=(100, 100)):
    label_to_image = {}

    # Loop through all images and store one random image path for each label
    for img_path, label in zip(image_paths, labels):
        if label not in label_to_image:
            label_to_image[label] = img_path

    unique_labels = list(label_to_image.keys())
    num_labels = len(unique_labels)

    cols = 3
    rows = math.ceil(num_labels / cols)  #grid size

    plt.figure(figsize=(15, 15))
    for i, label in enumerate(unique_labels):
        plt.subplot(rows, cols, i + 1)
        img_path = label_to_image[label]
        try:
            with Image.open(img_path) as img:
                img = img.resize(img_size)
                img_array = np.array(img)
            # Plot the image
            plt.imshow(img_array, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(label, fontsize=10)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
    plt.tight_layout()
    plt.show()

plot_one_image_per_label(images, labels)

# Print some statistics about the dataset
print(f"Total number of images: {len(images)}")
print(f"Number of unique labels: {len(set(labels))}")
print("Label distribution:")
for label in set(labels):
    print(f"{label}: {labels.count(label)}")

img_size = (100, 100)
X = []

for img_path in images:
    # Open the image file
    try:
        with Image.open(img_path) as img:
            img = img.resize(img_size)
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=-1)  #channel dimension
            X.append(img_array)   #processed images
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")

X = np.array(X)
print(X.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y = to_categorical(y_encoded)
print("Label classes:", label_encoder.classes_)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

X_train[1,:]

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.summary()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Loss: {loss}")

train_acc = history.history['accuracy']
train_loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

index_loss = np.argmin(val_loss)
index_acc = np.argmax(val_acc)

val_lowest = val_loss[index_loss]
val_highest = val_acc[index_acc]

Epochs = [i+1 for i in range(len(train_acc))]

loss_label = f'Best Epoch = {str(index_loss + 1)}'
acc_label = f'Best Epoch = {str(index_acc + 1)}'

plt.figure(figsize= (20,8))

plt.subplot(1,2,1)
plt.plot(Epochs , train_loss , label = 'Training Loss')
plt.plot(Epochs , val_loss , label = 'Validation Loss')
plt.scatter(index_loss +1 , val_lowest , s = 150 , c = 'blue' , label = loss_label)
plt.title('Training vs Validation (loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(Epochs , train_acc , label= 'Training Accuracy')
plt.plot(Epochs , val_acc , label = 'Validation Accuracy')
plt.scatter(index_acc + 1 , val_highest , s= 150 , c = 'blue' , label= acc_label)
plt.title('Training vs Validation (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show();

# Predict the labels for the test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1) #convert probabilities to class labels
true_labels = np.argmax(y_test, axis=1)  #covert encoded y_test to class labels

print("First few predictions:", predicted_labels[:10])
print("First few true labels:", true_labels[:10])


def plot_sample(ax, image, true_label, pred_label):
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"True: {label_encoder.classes_[true_label]}\nPred: {label_encoder.classes_[pred_label]}", fontsize=10)
    ax.axis('off')

num_samples = 10
num_columns = 3

num_rows = (num_samples + num_columns - 1) // num_columns  # number of rows
fig, axes = plt.subplots(num_rows, num_columns,figsize=(10,10))

axes = axes.flatten() # Flatten axes array for easier iteration

for i in range(num_samples):
    plot_sample(axes[i], X_test[i], np.argmax(y_test[i]), predicted_labels[i])

for j in range(num_samples, len(axes)):  # Remove unused subplots
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


