# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:48:29 2018

@author: Jintram
"""

MODELSAVEPATH = 'C:/Users/Jintram/Documents/Python_scripts/bacterial_segmentation/'
bac_image_path = 'C:/Users/Jintram/Temporary/ecoli_set1/'
x_folder = 'x_images/'
y_folder = 'y_images/'
    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

import datetime

from skimage.io import imread

import sklearn.preprocessing as pp

from keras.models import Sequential
import keras.backend as K

from keras.layers import Conv2D
#from keras.layers import Convolution2D
    # convolution for 2d
from keras.layers import MaxPooling2D 
    # pooling for 2d
from keras.layers import Flatten 
    # combine feature maps into 1 vector
from keras.layers import Dense 
    # conventional fully connected neural network that takes feature vector as input
from keras.losses import categorical_crossentropy
from keras.layers import Activation

# %% Import my data
import bacterial_data_preparation_batch
data_x, data_y = bacterial_data_preparation_batch.build_data(bac_image_path, x_folder, y_folder)

# images are expected in 3 dims, fix that
data_x_2 = np.expand_dims(data_x,axis=3)
# output should be three neurons binary, so fix that
data_y_2 = np.empty([len(data_y),3])
data_y_2[:] = 0
for i in range(len(data_y_2)):
    data_y_2[i,int(data_y[i])]=1

# %% create test and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_x_2, data_y_2, test_size = 0.2, random_state = 0)


# %% setting up a simple model for bacterial segmentation
# Note: this is *convolutional*, so 1st output of CNN are just transformed original images
    
bacmodel = Sequential()
# First convolutional layer 
bacmodel.add( Conv2D(6, (4,4), activation='relu', padding='same', input_shape=(21, 21, 1) ) )
bacmodel.add(MaxPooling2D(pool_size=(2,2)))

# Second convolutional layer
bacmodel.add( Conv2D(10, (4,4), activation='relu', padding='same', input_shape=(20, 20,1) ) )
bacmodel.add(MaxPooling2D(pool_size=(2,2)))

# Now flatten everything
bacmodel.add(Flatten())

# Very simple conventional network
bacmodel.add(Dense(output_dim = 8, activation = 'relu')) # hidden layer
bacmodel.add(Dense(output_dim = 8, activation = 'relu')) # hidden layer
bacmodel.add(Dense(output_dim = 3, activation = 'sigmoid')) # output layer

bacmodel.add(Activation('softmax'))

bacmodel.summary()

# %% Compile the model

# 
bacmodel.compile(optimizer='adam', loss='categorical_crossentropy')#, metric=['accuracy'])
    # for 1 output neuron, use binary_crossentropy
    # for >2 outcomes, use categorical_cross_entropy

# %% fit the model (training)

history = bacmodel.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 2000, epochs = 300, verbose=2)
#history = bacmodel.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=5, verbose=2)


# and save the model
now = datetime.datetime.now()
currenttimestr = now.strftime("%Y%m%d-%H%M")
bacmodel.save(MODELSAVEPATH + 'bacmodelfit_' + currenttimestr + '.h5')
bacmodel.save_weights(MODELSAVEPATH + 'bacmodelfit_' + currenttimestr + '_weights' + '.h5')

# %%
if 0:    

    # load previous model
    loadtimestr = '20180923-0727'
    bacmodel.load_weights(MODELSAVEPATH + 'bacmodelfit_' + loadtimestr + '_weights' + '.h5')

# %% plot model stats

plt.figure()
plt.plot(history.history['loss'])
plt.title('Loss function progression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# %% now, let's see how the model works
MARGIN_MASK=10

testimgpath='C:/Users/Jintram/Temporary/ecoli_set1/misc_images/cropped_2015-06-12_pos1crop-p-2-126.tif'
bacimg_x = np.array(imread(testimgpath))

#bac_imgs = os.listdir(bac_image_path_x)
#bacimg_x = np.array(imread(bac_image_path_x + bac_imgs[1]))


# feature scaling
bacimg_x = np.reshape(pp.scale( bacimg_x.flatten(), axis=0, with_mean=True, with_std=True, copy=True ),
                      np.shape(bacimg_x))

# %% analyze the image


img_shape = np.shape(bacimg_x)
output_img = np.empty([img_shape[0],img_shape[1],3])
output_img[:] = np.nan
#test_y = np.empty([img_shape[0],img_shape[1],21,21,1])
with tqdm(total=(img_shape[0]-2*MARGIN_MASK)) as bar:
    for idx_i in range(MARGIN_MASK+1,img_shape[0]-MARGIN_MASK):
        for idx_j in range(MARGIN_MASK+1,img_shape[1]-MARGIN_MASK):
            # 
            # get input
            current_x = bacimg_x[(idx_i-MARGIN_MASK):(idx_i+MARGIN_MASK+1), 
                                 (idx_j-MARGIN_MASK):(idx_j+MARGIN_MASK+1)]
            # adjust shape
            current_x_2 = np.expand_dims([current_x], axis=3)
            
            # fit    
            output_img[idx_i,idx_j] = bacmodel.predict(current_x_2)
         
        bar.update()

# %% select the maximum of the pixels
output_img_2 = np.empty([img_shape[0],img_shape[1],3])
output_img_2[:] = np.nan
for idx_i in range(MARGIN_MASK+1,img_shape[0]-MARGIN_MASK):
    for idx_j in range(MARGIN_MASK+1,img_shape[1]-MARGIN_MASK):

        output_img_2[idx_i,idx_j,np.argmax(output_img[idx_i,idx_j])] = 1
        
# %%
XLIM = [450,600]
YLIM = [230,80] 
        
final_loss = history.history['loss'][-1]


def axiscosmetics(ax):
    plt.xlim(XLIM), plt.ylim(YLIM) 
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])

fig=plt.figure()
fig.suptitle('Convolutional Neural network\n'+
             'CNN 6x(4x4), 10x(4x4) + DNN 8, 8, 3\n'+
             'Cross entropy loss = ' + '{0:.2f}'.format(final_loss))

ax=plt.subplot(131)
plt.imshow(bacimg_x,cmap='gray')
axiscosmetics(ax)
plt.title('Input image')

ax=plt.subplot(132)
plt.imshow(output_img)
axiscosmetics(ax)
plt.title('Prediction')

ax=plt.subplot(133)
plt.imshow(output_img_2)
axiscosmetics(ax)
plt.title('Prediction\n(highest selected)')

plt.show()





