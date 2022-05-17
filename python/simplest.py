from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2
from matplotlib import pyplot
import sys

def super_dummy_model(in_shape=(3,128,128)):
     model = Sequential()
     model.add(GaussianNoise(stddev=0.7))
     model.add(GaussianNoise(stddev=0.8))
     model.add(Dense(units=3))
     model.add(Dense(units=3))
     return model

print(sys.argv[1])
data = image_dataset_from_directory(sys.argv[1],batch_size=1,image_size=(128,128))
model = super_dummy_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

q=0
cap = cv2.VideoCapture(0)

"""while q != 27:
    _,img = cap.read()
    img = cv2.resize(img,(128,128))
    img = img/255 
    img=expand_dims(img,axis=0)
    predictions=model(img)
    cv2.imshow("noised",predictions.numpy()[0])
    q = cv2.waitKey(33)
"""
for imgi,label in data:
    img = imgi[0].numpy()/255
    cv2.imshow("original",img)
    orig = img
    img=expand_dims(img,axis=0)
    predictions=model(img)
    cv2.imshow("noised",predictions.numpy()[0])
    cv2.waitKey(0)
    break 
#model.build(input_shape=(1,3,128,128))
model.summary()
model.save("testa")
