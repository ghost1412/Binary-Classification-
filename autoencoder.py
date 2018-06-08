'''
		-------------------------
|---------------|  importing libraries    |------------------------------------------------------|
		--------------------------
'''
import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from statistics import mean
'''
		 -------------------------
|---------------|      reading 	csv	   |-------------------------------------------------------|
		 --------------------------
'''
reader = csv.reader(open("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data[1:25000]).astype("str")    #read data in string form
print(result.shape)
feature_names = result[0, 1 : len(result[1])-1]

#removing unwanted strings butrans = 0 and opana = 1 and empty cells with 0 and making 1st cell equal 0 so#
for j in range(0, len(result)):
  result[j][0] = '0'
  for i in range(0, len(result[j])):
    if(result[j][i] == "butrans"):
      result[j][i] = '7'
    if(result[j][i] == "opana"):
      result[j][i] = '8'
    if(result[j][i] == ''):
      result[j][i] = '0'
    if(result[j][i] == "Butrans and Opana"):
      result[j][i] = '9' 
    if(result[j][i] == 'Frequent'):
      result[j][i] = '0' 
    if(result[j][i] == 'Non Frequent'):
      result[j][i] = '1'

labels = result[1: , -2]                                      #label for classification
result_train = result[0: , 1:len(result[1])-2]              #training data for autoencoder
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
x_train = np.empty((len(result_train), 128, 128,1))            

for j in range(0, len(result_train)):                         #reshaping to 2d array for convolutional autoencoder
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1306), 'constant'),(128,128,1))
  x_train[j] = resultraintemp
'''
		 -------------------------
|---------------|    conv autoencoder	  |------------------------------------------------------|
		 --------------------------
'''
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K
input_img = Input(shape=(128, 128, 1)) 
'''
		 -------------------------
|---------------|        encoder	   |-------------------------------------------------------|
		 --------------------------
'''
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='con3')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same', name='max3')(x)
'''
	         -------------------------
|---------------|        decoder	  |--------------------------------------------------------|
		 -------------------------
'''
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

autoencoder = Model(input_img, decoded)
autoencoder.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='binary_crossentropy', metrics=['accuracy'])
'''
		 -------------------------
|---------------|    training model	  |-------------------------------------------------------|
              	 -------------------------
'''
from keras.callbacks import TensorBoard

x_test = x_train
history = autoencoder.fit(x_train, x_train,
               			 epochs= 200,
               			 batch_size=10,
              		         shuffle=True,
                		 validation_data=(x_test, x_test),
               			 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

model_json = autoencoder.to_json()
# serialize weights to HDF5
decoded_imgs = autoencoder.predict(x_test)
print(mean(history.history['acc']))
print(mean(history.history['val_acc']))
with open("model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("model.h5")



