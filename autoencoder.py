######################data preperation##################################################
import numpy as np
import pandas as pd
import io
import csv
from numpy import empty

#######################reading csv######################################################

reader = csv.reader(open("/home/tonystark/Downloads/abc2class.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data).astype("str")    #read data in string form
feature_names = result[0, 1 : len(result[1])-1]

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
labels = result[1: , -1]                                      #label for classification
result_train = result[1: , 1:len(result[1])-1]              #training data for autoencoder
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
x_train = np.empty((len(result_train), 128, 128,1))            

for j in range(0, len(result_train)):                         #reshaping to 2d array for convolutional autoencoder
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1304), 'constant'),(128,128,1))
  x_train[j] = resultraintemp
  
#########################autoencoder#####################################

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K


################encoder##############################################

input_data = Input(shape=(128, 128, 1)) 
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(input_data)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='con3')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same', name='max3')(x)

##############decoder################################################

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

##############training################################################

from keras.callbacks import TensorBoard

x_test = x_train
autoencoder.fit(x_train, x_train,
                epochs= 4,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


###########saving weights################################################

model_json = autoencoder.to_json()
# serialize weights to HDF5
decoded_imgs = autoencoder.predict(x_test)
#print(x_test[1000])
#print(decoded_imgs[1000])
with open("model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("model.h5")
#print(newData)  




