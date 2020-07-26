#!/usr/bin/env python
# coding: utf-8

# ## A deep learning classifier to classify the fashion items in 10 respective classes in the Fashion MNIST dataset.

# ### Importing all required libraries

# In[1]:


import os
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten


# ### Setting all the paths

# In[2]:


data_path=os.getcwd()+"\\dataset"                       # get the current working directory and add the dataset path to it 

# Setting the trainig set path 
train_path = data_path+'\\train-images-idx3-ubyte'           
train_label_path = data_path+'\\train-labels-idx1-ubyte'

# Setting the test set path
test_path = data_path+'\\t10k-images-idx3-ubyte'
test_label_path = data_path+'\\t10k-labels-idx1-ubyte'


# ### Now we need to convert the idx files to numpy arrays inorder to process the images

# In[3]:


trainX = idx2numpy.convert_from_file(train_path)
trainY = idx2numpy.convert_from_file(train_label_path)

testX = idx2numpy.convert_from_file(test_path)
testY = idx2numpy.convert_from_file(test_label_path)


# ### Just checking whether the images are properly converted

# In[4]:


for i in range(6):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()


# ### Checking the shape of all the arrays , to get a clear idea of the amount of data to be processed 

# In[5]:


print("shape of trainX: "+str(trainX.shape))
print("shape of trainY: "+str(trainY.shape))
print("shape of testX: "+str(testX.shape))
print("shape of testY: "+str(testY.shape))


# ### Reshaping the images
# 
# We know that the images are all pre-segmented (e.g. each image contains a single item of clothing), that the images all have the same square size of 28×28 pixels, and that the images are grayscale. Therefore, we can load the images and reshape the data arrays to have a single color channel.

# In[6]:


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))


# ### Scaling Pixels
# The pixel values for each image in the dataset are unsigned integers in the range between black and white, or 0 and 255.
#  
# To normalize the pixel values of grayscale images we first convert the data type from unsigned integers to floats, then divide the pixel values by the maximum value.

# In[7]:


#scaling of pixels
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255


# ### One Hot Encoding
# 
# There are 10 classes and that classes are represented as unique integers.
# So we use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value. We can achieve this with the to_categorical() utility function.

# In[8]:


trainY = to_categorical(trainY)
testY = to_categorical(testY)


# ### Preparing the model
# Steps to create Deep Neural Network Classifier
# *******************
# <ol>
#     <li> Initialization </li>
#     <li> Convolution </li>
#     <li> Max pooling </li>
#     <li> Flattening </li>
#     <li> Full connection </li>
# </ol>
# **********************
# <p>1) Input Shape : 28 , 28 , 1</p>
# <p>2) Activation function used in Convolution layer : Rectified linear unit(ReLu) i.e, y = max(0, x)</p>
# <p>3) Activation function used in Dense output layer : Softmax ( softmax is useful because it converts the output of the last layer in a neural network into what is essentially a probability distribution)</p>
# <p>4) Optimizer : Adam</p>
# <p>5) Loss function : Categorical Cross Entropy ( This is because we have more than 2 categorical features i.e, total of 10 )</p>

# In[9]:


def DNN_classifier():
    model=Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ### Training the model
# <p>A total of 12 epochs is done with a batch size of 32</p>
# <p>Verbose is set to 1 so that we can view the progress bar along with time taken in 1 epoch, loss and accuracy.</p>

# In[14]:


model=DNN_classifier()
model.fit(trainX, trainY, epochs=20, batch_size=32, validation_data=(testX, testY), verbose=1)
model.save('final_model.h5') # model is saved to an h5 file for further usage 


# ### Testing the model with some sample images
# 
# Classes:
# <ol start="0">
#     <li>T-shirt/top</li>
#     <li>Trouser</li>
#     <li>Pullover</li>
#     <li>Dress</li>
#     <li>Coat</li>
#     <li>Sandal</li>
#     <li>Shirt</li>
#     <li>Sneaker</li>
#     <li>Bag</li>
#     <li>Ankle boot</li>
# </ol>

# In[15]:


# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class
def run_image_classifier(image_path):
    # load the image
    img = load_image(image_path)
    # load model
    model = load_model('final_model.h5')
    # predict the class
    result = model.predict_classes(img)
    n = result[0]
    print(result)
    print("Model Prediction is: ",end="")
    if n == 0:
        print("T-shirt/top")
    elif n == 1:
        print("Trouser")
    elif n == 2:
        print("Pullover")
    elif n == 3:
        print("Dress")
    elif n == 4:
        print("Coat")
    elif n == 5:
        print("Sandal")
    elif n == 6:
        print("Shirt")
    elif n == 7:
        print("Sneaker")
    elif n == 8:
        print("Bag")
    elif n == 9:
        print("Ankle Boot")



# Testing sample image 1
run_image_classifier('sample_image.png')


# checking the input image
plt.imshow(load_img('sample_image.png'), cmap=plt.cm.binary)


# In[16]:


# Testing sample image 2
run_image_classifier('sample_image2.jpg')


# checking the input image 2
plt.imshow(load_img('sample_image2.jpg'), cmap=plt.cm.binary)


# In[17]:


# Testing sample image 3
run_image_classifier('sample_image3.jpg')


# checking the input image 3
plt.imshow(load_img('sample_image3.jpg'), cmap=plt.cm.binary)

