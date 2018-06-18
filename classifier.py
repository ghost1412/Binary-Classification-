'''
		-------------------------
---------------|  importing libraries    |-------------------------------------------------------
		--------------------------
'''
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K
from statistics import mean
import numpy as np
import pandas as pd
import io
import csv
from numpy import empty
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
'''
		-------------------------
|---------------|        encoder	 |-------------------------------------------------------|
		--------------------------
'''
input_img = Input(shape=(128, 128, 1)) 
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max1')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same', name='max2')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same', name='max3')(x)
'''
		-------------------------
|---------------|        decoder	 |--------------------------------------------------------|
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

'''
		-------------------------------------------------------------------------
|---------------| loading pretrained weights of autoencoder and freezing autoenc layers	 |-------------------------|
		|			transferlearning				 |	 
		-------------------------------------------------------------------------
'''
autoencoder.load_weights("model.h5")
print("Loaded Model from disk")
print(encoded[1])
autoencoder.layers[0].trainable = False
autoencoder.layers[1].trainable = False
autoencoder.layers[2].trainable = False
autoencoder.layers[3].trainable = False
autoencoder.layers[4].trainable = False
autoencoder.layers[5].trainable = False
autoencoder.layers[6].trainable = False
autoencoder.layers[7].trainable = False
autoencoder.layers[8].trainable = False
#autoencoder.layers[9].trainable = False
#autoencoder.layers[10].trainable = False
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
autoencoder.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='binary_crossentropy')
autoencoder.summary()
'''
		---------------------------------
|---------------|     classifier's layers	 |--------------------------------------------------------|
		---------------------------------
'''

y = Conv2D(32, (3, 3), padding = 'same', activation='relu' , name = 'conv8')(encoded)
y = Conv2D(32, (3, 3), activation='relu')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2), name = 'max7')(y)
y = Dropout(0.25)(y)   
#y = Flatten()(y)
y = Conv2D(64, (3, 3), padding='same', activation='relu', name = 'conv9')(y)
y = Conv2D(64, (3, 3), activation='relu')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(0.25)(y)   
y = Conv2D(64, (3, 3), padding='same',activation='relu', name = 'conv10')(y)
#y = Conv2D(64, (3, 3), activation='relu')(y)
y = BatchNormalization()(y)
y = MaxPooling2D(pool_size=(2, 2), name = 'max8')(y)
y = Dropout(0.25)(y)
y = Flatten()(y)
y = Dense(512, activation='relu')(y)
y = Dropout(0,5)(y)
y1 = Dense(2, activation='sigmoid')(y)

classifier= Model(inputs=autoencoder.input, outputs=y1)		#input to autoencoder input and output at y1

'''
		-------------------------
|---------------|     preparing data	 |--------------------------------------------------------|
		-------------------------
'''
from sklearn.model_selection import train_test_split

reader = csv.reader(open("/home/tonystark/Downloads/gh/abc2classwoenrolid.csv", "r"), delimiter=",")
data = list(reader)
result = np.array(data[1:]).astype("str")    #read data in string form
print(result.shape)

#removing unwanted strings butrans = 0 and opana = 1 and empty cells with 0 and making 1st cell equal 0 so#
for j in range(0, len(result)):
  result[j][0] = '0'
  for i in range(0, len(result[j])):
    if(result[j][i] == "butrans"):
      result[j][i] = '8'
    if(result[j][i] == "opana"):
      result[j][i] = '9'
    if(result[j][i] == ''):
      result[j][i] = '0'
    if(result[j][i] == "Butrans and Opana"):
      result[j][i] = '7' 
    if(result[j][i] == 'Frequent'):
      result[j][i] = '0' 
    if(result[j][i] == 'Non Frequent'):
      result[j][i] = '1'
labels = result[1: , -1]                                      #label for classification
one_hot_labels = to_categorical(labels, num_classes=2)
result_train = result[1: , 1:len(result[1])-2]                #sliceing last and 1st coloumn out
#converting array to float type after removing string
result_train = np.array(result_train).astype(np.float)        #converting data to float
input = np.empty((len(result_train), 128, 128,1))             #features

for j in range(0, len(result_train)):                         #input to classifer to 128*128
  resultraintemp = np.reshape(np.pad(result_train[j], (0, 1306), 'constant'),(128,128,1))
  input[j] = resultraintemp

#train_labels_one_hot  = np.empty((len(result), 2, 1,))
#hot_encoding = np.zeros((2, 1))
#for i in range(0, len(result)):
 #   if(result[i][-1] == 1):
  #      hot_encoding[1] = 1
  #  else:
   #     hot_encoding[0] = 0
   # train_labels_one_hot[i] = hot_encoding 

'''
		-------------------------
|---------------|  training classifier	 |--------------------------------------------------------|
		-------------------------
'''
from keras.callbacks import TensorBoard

x_train, x_test, y_train, y_test = train_test_split(input,
                                                          one_hot_labels,
                                                          test_size=0.33,
                                                          random_state=42)

classifier.compile(SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()
history = classifier.fit(x_train, y_train,
                	 epochs=100,
               		 batch_size=6,
               		 shuffle=True,
               		 validation_data=(x_test, y_test),
               		 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
classifier.predict(x_test)
print(mean(history.history['acc']))
print(mean(history.history['val_acc']))

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot(history.history['acc'], history.history['val_acc'])
fig.savefig('/home/tonystark/Downloads/gh/')   # save the figure to file
plt.close(fig)    # close the figure
