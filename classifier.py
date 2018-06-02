
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K

################encoder#########################################################

input_data = Input(shape=(128, 128, 1)) 
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(input_data)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same', name='max3')(x)

##############decoder############################################################

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2), name='up1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv5')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2), name='up2')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv6')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2),name='up3')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='conv7')(x)

autoencoder = Model(input_data, decoded)
autoencoder.compile(RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0), loss='binary_crossentropy')

#############loading pre-tranined weights to autoencoder###################################

autoencoder.load_weights("model.h5")
print("Loaded Model from disk")

################transfer learning freezing encoder part and removing decoder part##############

autoencoder.layers[0].trainable = False
autoencoder.layers[1].trainable = False
autoencoder.layers[2].trainable = False
autoencoder.layers[3].trainable = False
autoencoder.layers[4].trainable = False
autoencoder.layers[5].trainable = False
autoencoder.layers[6].trainable = False
autoencoder.layers[7].trainable = False
autoencoder.layers[8].trainable = False
autoencoder.layers[9].trainable = False
autoencoder.layers[10].trainable = False
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()

############transfered learning from encoder to classifier by joining classifier's input layer to encoder output#########

import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

y = Conv2D(32, (3, 3), padding = 'same', activation='relu')(encoded)	#classifier's layers
y = MaxPooling2D(pool_size=(2, 2))(y)
#y = Dropout(0.25)(y)   
#y = Flatten()(y)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
y = Conv2D(64, (3, 3), activation='relu')(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Flatten()(y)
#y = Dropout(0.25)(y)
y1 = Dense(1, activation='softmax')(y)

classifier= Model(inputs=autoencoder.input, outputs=y1)		#input to autoencoder input and output at y1

###########preparing data#################################################################

from sklearn.model_selection import train_test_split

reader = csv.reader(open("/home/tonystark/Downloads/abc2class.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data).astype("str")    #read data in string form

#removing unwanted strings butrans = 0 and opana = 1 and empty cells with 0 and making 1st cell equal 0 so#
for j in range(0, len(result)):
  result[j][0] = '0'
  for i in range(0, len(result[j])):
    if(result[j][i] == "butrans"):
      result[j][i] = '0'
    if(result[j][i] == "opana"):
      result[j][i] = '1'
    if(result[j][i] == ''):
      result[j][i] = '0'

labels = result[1: , -1]                                      #label for classification 0 or 1
result_train = result[1: , 1:len(result[1])-1]                #sliceing last and 1st coloumn out
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
input = np.empty((len(result_train), 128, 128,1))             #features

for j in range(0, len(result_train)):                         #input to classifer to 128*128
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1304), 'constant'),(128,128,1))
  input[j] = resultraintemp

#train_labels_one_hot  = np.empty((len(result), 2, 1,))
#hot_encoding = np.zeros((2, 1))
#for i in range(0, len(result)):
 #   if(result[i][-1] == 1):
  #      hot_encoding[1] = 1
  #  else:
   #     hot_encoding[0] = 0
   # train_labels_one_hot[i] = hot_encoding 

###################training classifier###########################################################
from keras.callbacks import TensorBoard

x_train, x_test, y_train, y_test = train_test_split(input,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = classifier.fit(x_train, y_train,
                		 epochs=50,
               			 batch_size=128,
               			 shuffle=True,
               			 validation_data=(x_test, y_test),
               			 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
classifier.predict(x_test)

