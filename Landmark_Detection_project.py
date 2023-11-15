#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import os
import random
from PIL import Image
import keras


# pip install keras

# pip install tensorflow

# pip install -U tensorflow_addons

# In[2]:


# Import DataSet

df = pd.read_csv("dataset/train.csv")
base_path='/images/'


# In[3]:


df


# In[4]:


# Filtering DataFrame and calculation of unique Landmark Classes

samples = 20000

df=df.loc[df['id'].str.startswith('00',na=False),:]
num_classes = len(df['landmark_id'].unique())
num_data = len(df)
num_data


# In[5]:


# Number of unique landmark classes in the filtered DataFrame.

num_classes


# In[6]:


# Creation of DataFrame

data = pd.DataFrame(df['landmark_id'].value_counts())
data.reset_index(inplace=True)


# In[7]:


data


# In[8]:


# Assignment of Column names

data.columns=['landmark_id','count']


# In[9]:


# Summary Statistics

data['count'].describe()


# In[10]:


# Creation of Histogram Plot

plt.hist(data['count'],100,range=(0,32),label='test')


# In[11]:


plt.hist(df["landmark_id"], bins=df["landmark_id"].unique())


# In[12]:


# Fitting of encoder to unique values of landmark_id column

from sklearn.preprocessing import LabelEncoder
lencoder=LabelEncoder()
lencoder.fit(df['landmark_id'])


# In[13]:


df


# In[14]:


# Transform the label(s) into numerical values.

def encoder_label(lbl):
    return lencoder.transform(lbl)


# In[15]:


# Numerical value(s) back into their original categorical labels.

def decoder_label(lbl):
    return lencoder.inverse_transform(lbl)


# In[16]:


# Function to retrieve images and labels from a DataFrame based on an index.

def get_image_from_num(num,df):
    fname,label=df.iloc[num,:]
    fname=fname+'.jpg'
    f1=fname[0]
    f2=fname[1]
    f3=fname[2]
    path=os.path.join(f1,f2,f3,fname)
    im=cv2.imread(os.path.join(base_path,path))
    return im,label


# In[17]:


# Displaying four randomly selected images 

print('4 sample images')
fig = plt.figure(figsize = (16,16))
for i in range(1,5):
    ri=random.choices(os.listdir(base_path),k=3)
    folder = base_path + ri[0] + '/'  + ri[1] +  '/' +  ri[2]

    random_img = random.choice(os.listdir(folder))
    img = np.array(Image.open(os.path.join(folder, random_img)))
    fig.add_subplot(1,4,i)
    plt.imshow(img)
    plt.axis('off')
plt.show()


# In[18]:


# Building of the model

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras import Sequential


# In[19]:


# Hyperparameter defintiions

learning_rate=0.001
decay_speed=1e-6
momentum=0.09
loss_function="spares_categorial_crossentropy"
source_model=VGG19(weights=None)
drop_layer=Dropout(0.5)
drop_layer2=Dropout(0.5)


# In[20]:


# Building a Sequential Neural Network

model=Sequential()
for layer in source_model.layers[:-1]:
    if  layer == source_model.layers[-25]:
        model.add(BatchNormalization())
    model.add(layer)
model.add(Dense(num_classes,activation='softmax'))


# In[21]:


# Summary of Architecture and parameters

model.summary()


# In[22]:


# Compiling the neural network model using the Adam optimizer with specified settings for loss and accuracy metrics.

from tensorflow.keras.optimizers import RMSprop

optim1 = RMSprop(learning_rate=0.001)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[23]:


# Creating a function to resize images using OpenCV, handling exceptions and providing an error message if resizing fails.

def image_resize(im,target_size):
    try :
        return cv2.resize(im,target_size)
    except Exception as e:
        print("Error resizing image:", str(e))
        return None


# In[24]:


# Function to retrieve a batch of resized images and encoded labels from a DataFrame.

def get_batch(dataframe,start,batch_size):
    image_array=[]
    label_array=[]

    last_img=start+batch_size
    if(last_img)>len(dataframe):
        last_img=len(dataframe)

    for idx in range(start,last_img):
        im,label=get_image_from_num(idx,dataframe)
        im=image_resize(im, (224,224)) / 255.0

        image_array.append(im)
        label_array.append(label)
        
    label_array=encoder_label(label_array)
    
    return np.array(image_array), np.array(label_array)


# In[25]:


# Splitting the DataFrame into training and validation sets with an 80-20 ratio for training and validation data, 
# respectively, based on a random shuffle.

batch_size=16
epoch_shuffle=True
weights_classes=True
epochs=1

train, val=np.split(df.sample(frac=1), [int(0.8*len(df))])


# In[26]:


len(train)


# In[27]:


len(val)


# In[28]:


# Training a model over a specified number of epochs, optionally shuffling data in each epoch and
# saving the trained model to a file.
for e in range(epochs):
    print('Epoch:' + str(e+1) + '/' + str(epochs))
    
    if epoch_shuffle:
        train = train.sample(frac = 1)
    for it in range(int(np.ceil(len(train)/batch_size))):
        X_train, Y_train = get_batch(train, it*batch_size, batch_size)
        
        model.train_on_batch(X_train, Y_train)
model.save('Model')


# In[29]:


# Test
# Evaluating model predictions on validation data, categorizing errors, and storing results with batch size 16.

batch_size=16
errors=0
good_preds=[]
bad_preds=[]

for it in range(int(np.ceil(len(val)/batch_size))):
    X_val,Y_val = get_batch(val,it*batch_size,batch_size)

    result=model.predict(X_val)
    cla=np.argmax(result,axis=1)
    for idx,res in enumerate(result):
        if cla[idx]!=Y_val[idx]:
            errors=errors+1
            bad_preds.append([batch_size*it+idx,cla[idx],res[cla[idx]]])
        else:
            good_preds.append([batch_size*it+idx,cla[idx],res[cla[idx]]])


# In[30]:


# Converting 'good_preds' to abeda NumPy array and sorting it based on prediction scores in descending order.

good_preds=np.array(good_preds)
good_preds=sorted(good_preds,key=lambda x:x[2],reverse=True)


# In[31]:


# Displaying the top 5 images with the highest prediction scores, along with their labels and
# sample counts in the corresponding classes.

fig=plt.figure(figsize=(16,16))
for i in range(4):
    n=int(good_preds[i][0])
    img,lbl=get_image_from_num(n,val)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig.add_subplot(1,5,i+1)
    plt.imshow(img)
    lbl2=np.array(int(good_preds[i][1]))
    sample_cnt = list(df.landmark_id).count(lbl)
    plt.title('label:' + str(lbl) + '\nclassified as:' + str(lbl2) + '\nsamples in class ' + 
              str(lbl) + ':' + str(sample_cnt))
    plt.axis('off')
plt.show()


# In[32]:


fig = plt.figure(figsize=(16, 16))
num_images_to_display = min(5, len(good_preds)) 

for i in range(num_images_to_display):
    n = int(good_preds[i][0])
    img, lbl = get_image_from_num(n, val)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig.add_subplot(1, 5, i + 1)
    plt.imshow(img)
    lbl2 = int(good_preds[i][1])
    sample_cnt = list(df.landmark_id).count(lbl)
    plt.title('label:' + str(lbl) + '\nclassified as:' + str(lbl2) + '\nsamples in class ' +
              str(lbl) + ':' + str(sample_cnt))
    plt.axis('off')

plt.show()


# In[33]:


fig=plt.figure(figsize=(16,16))
for i in range(5):
    n=int(good_preds[i][0])
    img,lbl=get_image_from_num(n,val)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig.add_subplot(1,5,i+1)
    plt.imshow(img)
    lbl2=np.array(int(good_preds[i][1]))
    sample_cnt = list(df.landmark_id).count(lbl)
    plt.title('label:' + str(lbl) + '\nclassified as:' + str(lbl2) + '\nsamples in class ' + 
              str(lbl) + ':' + str(sample_cnt))
    plt.axis('off')
plt.show()


# In[ ]:




