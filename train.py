from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
from keras.datasets import mnist
#from keras.utils import to_catagorical
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
y_test = np_utils.to_categorical(y_test, 10)
y_train = np_utils.to_categorical(y_train, 10)

# design model
model = Sequential()
model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Convolution2D(50, (5, 5)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=0.001)
# compile model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# training model
model.fit(x_train, y_train, batch_size=100, epochs=5)
# test model
print (model.evaluate(x_test, y_test, batch_size=100))
# save model
model.save('/Users/zhuwentao/Desktop/mnist/mnist.h5')
