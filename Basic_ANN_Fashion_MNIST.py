#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Import dataset
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
#Visualize
import matplotlib.pyplot as plt


# In[15]:


#Load inbulit dataset
fmnist = tf.keras.datasets.fashion_mnist


# In[16]:


#Splitting
(X_train_fashion, y_train_fashion),(X_test_fashion, y_test_fashion) = fmnist.load_data()


# In[17]:


#Check shape and size
print("X train: ",X_train_fashion.shape)
print("Y train: ",y_train_fashion.shape)
print("X test: ",X_test_fashion.shape)
print("Y test: ",y_test_fashion.shape)


# In[18]:


#Display few images
plt.figure(figsize=(10,10))
for i in range(15):
  plt.subplot(5,5,i+1)
  plt.imshow(X_train_fashion[i],cmap='binary')
  plt.axis("off")
plt.show()


# In[19]:


#Normalize
X_train_fashion = X_train_fashion/255
X_test_fashion = X_test_fashion/255


# In[20]:


#Model Building
#ANN Architecture
fashion_model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(24,activation='relu'),
     tf.keras.layers.Dense(32,activation='softmax')]
)
#Compile and build
fashion_model.compile(optimizer = tf.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy')


# In[21]:


#Fit
fashion_model.fit(X_train_fashion, y_train_fashion, epochs=5)


# In[22]:


#Display performance
fashion_model.summary()


# In[23]:


#Testing
fashion_model.evaluate(X_test_fashion,y_test_fashion)


# In[24]:


classifications = fashion_model.predict(X_test_fashion)

plt.imshow(X_test_fashion[70])
result = np.where(classifications[70] == max(classifications[70]))

print(f"\nOur prediction: {result[0][0]}")
print(f"\nActual answer: {y_test_fashion[70]}")


# In[ ]:




