import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open('best_model.sav', 'rb') as pick:
    model = pickle.load(pick)

with open('test_data.pickle', 'rb') as pick:
    data = pickle.load(pick)

file_names = []
test_data = []

for feature, file_name in data:
    test_data.append(feature)
    file_names.append(file_name)

scaler = StandardScaler().fit(test_data)
X_test_scaled = scaler.transform(test_data)

n_components = 980

pca = PCA(n_components=n_components)
X_test_pca = pca.fit_transform(X_test_scaled)

random_index = random.randint(0, len(X_test_pca) - 1)
random_image_pca = X_test_pca[random_index].reshape(1, -1) 
random_file_name = file_names[random_index]

prediction = model.predict(random_image_pca)

categories = ['Cat', 'Dog']
predicted_label = categories[prediction[0]]

print(f'File: {random_file_name} - Prediction: {predicted_label}')

original_image = np.array(test_data[random_index]).reshape(64, 64, 3) 
plt.title(f'Prediction: {predicted_label}')
plt.show()
