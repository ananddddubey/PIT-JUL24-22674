import os
import numpy as np
import cv2
import time
import pickle

start_time = time.time()

dir = r"C:\Users\jidub\OneDrive\Documents\internship\Prodigy Infotech\Task 3\dogs-vs-cats\train\train"

# Check if the directory exists
if not os.path.exists(dir):
    print(f"Directory does not exist: {dir}")
else:
    file = os.listdir(dir)

    train_cat_dir = []
    train_dog_dir = []

    for f in file:
        target = f.split(".")[0]
        full_path = os.path.join(dir, f)

        if target == "cat":
            train_cat_dir.append(full_path)

        if target == "dog":
            train_dog_dir.append(full_path)

    IMG_SIZE = (64, 64)

    data = []
    labels = []

    def process_images(file_list, label):
        for file_path in file_list:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is not None:
                img_resized = cv2.resize(img, IMG_SIZE)
                img_normalized = img_resized / 255.0
                img_flattened = np.array(img_normalized).flatten()
                data.append([img_flattened, label])

    process_images(train_cat_dir, 0)
    process_images(train_dog_dir, 1)

    with open('data1.pickle', 'wb') as pick_in:
        pickle.dump(data, pick_in)

    print(f"Saving Time: {time.time() - start_time} seconds")
