#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model


# In[30]:


get_ipython().system('pip install pydot')


# In[9]:


#Loading the data
((x_train, y_train), (x_test, y_test)) = mnist.load_data()


# In[10]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[11]:


num_labels = len(np.unique(y_train))
num_labels


# In[12]:


y_train = to_categorical(y_train, num_classes=num_labels)
y_test = to_categorical(y_test, num_classes=num_labels)


# In[15]:


#Reshaping and Normalizing the data
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# In[18]:


#Parameters
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32


# In[19]:


#Creating the left branch of the Y-Network
left_input = Input(shape = input_shape)
x = left_input
filters = n_filters 

for _ in range(3):
    x = Conv2D(filters = filters, kernel_size= kernel_size, padding='same', activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    
    filters = filters*2


# In[23]:


#Creating the right branch of the Y-Network
right_input = Input(shape = input_shape)
y = right_input
filters = n_filters 

for _ in range(3):
    y = Conv2D(filters = filters, kernel_size= kernel_size, padding='same', activation='relu', dilation_rate=2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    
    filters = filters*2


# In[24]:


#Merging left and right branches
y = concatenate([x,y])
y = Flatten()(y)
y = Dropout(dropout)(y)
output = Dense(num_labels, activation='softmax')(y)


# In[27]:


model = Model([left_input, right_input], output)
#plot_model(model, to_file='y-network.png', show_shapes=True)
model.summary()


# In[31]:


plot_model(model, to_file='y-network.png', show_shapes=True)


# In[33]:


model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# In[ ]:


model.fit([x_train, x_train],
          y_train,
          validation_data = ([x_test, x_test], y_test),
          epochs = 20,
          batch_size=batch_size)


# In[ ]:




