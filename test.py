from keras.models import load_model
import numpy as np
from PIL import Image


def ImageToMatrix(filename):
    im = Image.open(filename)
    im = im.resize((28,28))
    #change to greyimage
    im=im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='int')
    return data

model = load_model('/Users/zhuwentao/Desktop/mnist/mnist.h5')


while 1:
    filename = input('Filename: ')
    data = ImageToMatrix('/Users/zhuwentao/Desktop/mnist/'+ filename +'.png')
    data = np.array(data)
    data = data.reshape(1, 28, 28, 1)
    print (model.predict_classes(data, batch_size=1, verbose=0))
    choice = input('Continue or not(y or n): ')
    if choice == 'n':
        break
