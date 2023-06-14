import keras,os
from keras.models import Sequential

#Sequential from keras.models gets our neural network as sequential network
# it can be sequential layer or graph

from keras.layers import Conv2D
# worling with images which are 2d genreally
# use 3d if working with videos

from keras.layers import MaxPooling2D

#avg pooling , sum pooling and max pooling etc

from keras.layers import Flatten

#we'll convert the 2d image vector to a flattened array of features

from keras.layers import Dense
# connecting neural layers

from keras.preprocessing.image import ImageDataGenerator
# this helps in resclaing , rotating zooming flipping etc

import numpy as np



#import data and stuff using code otherwise refer to vid.   https://youtu.be/bEsRLXY7GCo

VGG=keras.applications.VGG16(input_weights=(224,224,3), include_top=False, weights='imagenet')

# all the layers are present inside this feautre extractor vgg16 is inbuilt

VGG.trainable=False
#we dont want to train the first 13 cnn layers , we are using pre decided weights from imagenet
# we are only training the last two layers

model=keras.Sequential([
    VGG,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activcation="relu"),
    keras.layers.Dense(units=256,activcation="relu"),
    keras.layers.Dense(units=2,activation="softmax")
])
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()


hist=model.fit_generator(steps_per_epoch=100,generator=, validation_data=, validation_steps=10,epoch=5)
model.save('vgggclf.h5')


#visualizing all the our params

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["accuracy, validation accuracy, loss , vaclidation loss"])
plt.show()




#testing

from keras.models import load_model
saved_model=load_model("vggclf.h5")
output=saved_model.predict(img)
if output[0][0]>output[0][1]:
    print("cat")

else:
    print("dog")



#    ######### check the traditional and coller way of doing it #######