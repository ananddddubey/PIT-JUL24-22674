import pickle
import random
import os
import numpy as np
import cv2
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time

start_time = time.time()

with open('data1.pickle', 'rb') as pick_in:
    data = pickle.load(pick_in)

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


pca = PCA(n_components=980)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


with open('best_model.sav', 'rb') as pick:
    model = pickle.load(pick)

model_score = model.score(X_test_pca, y_test)

print('Model score: ', model_score)

predictions = model.predict(X_test_pca)

categories = ['Cat', 'Dog']
print(classification_report(y_test, predictions, target_names=categories))

random_indices = random.sample(range(len(X_test_pca)), 6)

plt.figure(figsize=(10, 5))
for i, random_index in enumerate(random_indices):
    random_image = X_test[random_index]
    random_image_pca = X_test_pca[random_index].reshape(1, -1)

    predicted_label = model.predict(random_image_pca)[0]

    predicted_category = categories[predicted_label]
    actual_category = categories[y_test[random_index]]

    reshaped_image = np.array(random_image).reshape(50, 50, 3)

    plt.subplot(2, 3, i + 1)
    plt.imshow(reshaped_image)
    plt.title(f"Prediction: {predicted_category}\nActual: {actual_category}")
    plt.axis('off')

plt.tight_layout()
plt.show()
