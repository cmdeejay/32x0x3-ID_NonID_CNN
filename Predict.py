import cv2
import keras
import matplotlib.pyplot as plt
categories = ["ID FRONT", "ID BACK", 'Non_ID']


def prepare(path):
    img_size = 80
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)


image_dir = "C:/Users/charmy.chen/Desktop/Client/3.jpg"

# img_array = cv2.imread(image_dir)
# new_array = cv2.resize(img_array, (80, 80))
# plt.imshow(new_array)
# plt.show()

model = keras.models.load_model('32x0x3-ID_NonID_CNN.model')
prediction = model.predict([prepare(image_dir)]).tolist()
print(categories[prediction[0].index(1)])
