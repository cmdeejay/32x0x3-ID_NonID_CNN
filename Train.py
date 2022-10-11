from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import TensorBoard
import pickle
import time

X = pickle.load(open('x.pickle', 'rb'))
Y = pickle.load(open('y.pickle', 'rb'))

X = X/255.0
dense_layers = [0, 1, 2, 3]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"ID_NonID-{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f'logs/ID_NonID/{NAME}')
            model = Sequential()
            model.add(Conv2D(kernel_size=layer_size, strides=(3, 3), input_shape=X.shape[1:], activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(kernel_size=layer_size, strides=(3, 3), activation='relu'))
                model.add(MaxPool2D(pool_size=(2, 2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size, activation='relu'))

            model.add(Dense(3, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X, Y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[tensorboard])
            model.save(f'{layer_size}x{dense_layer}x{conv_layer}-ID_NonID_CNN.model')
