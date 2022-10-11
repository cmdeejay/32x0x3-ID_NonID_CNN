import numpy as np
import os
import cv2
import random
import pickle

data_dir = r"C:/Users/charmy.chen/Desktop/ID ML/ID"
categories = ["ID FRONT", "ID BACK", "Non_ID"]
training_data = []


def create_training_data():
    x = []
    y = []
    for category in categories:
        img_size = 150  # Resize the image to standardise the data metrix
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Using cv2 to read images in Grayscale.
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])  # Zip category and the features into list.
            except Exception as e:
                pass
        random.shuffle(training_data)
    for features, label in training_data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, img_size, img_size, 1)
    y = np.array(y)
    pickle_out = open('x.pickle', 'wb')  # Data Serialization
    pickle.dump(x, pickle_out)
    pickle_out.close()
    pickle_out = open('y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()
    pickle_in = open('x.pickle', 'rb')
    x = pickle.load(pickle_in)
    return x
