#!/usr/bin/env python
# coding: utf-8

# In[1]:


# task 2 - SVM


# In[2]:


get_ipython().run_cell_magic('bash', '', '# pip3 install tensorflow # please uncomment for first time\n# pip3 install keras\n# pip3 install seaborn')


# In[3]:


# Import the needed packages
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# For CNN layers and model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# For SVC
from sklearn.svm import SVC


# dont show warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# declare glabal variables
(x_train, y_train, x_test, y_test) = [0 ,0 ,0 , 0]
model = False


# In[5]:


def load_data() :
    # Get mnist data set and split to train and test
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(type(x_train))
    


# In[ ]:





# In[6]:


def pre_process_data() :
    # Reshape the datasets from 3 dim to 4 dim - required
    global x_train, y_train, x_test, y_test

    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    


# In[7]:


def normalize_data() :
    # Convert to float
    global x_train, y_train, x_test, y_test

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the RGB codes - Divide by 255
    x_train /= 255
    x_test /= 255
    x_train.shape
    


# In[ ]:





# In[8]:


def create_model():
    # Create model
    input_shape = (28, 28, 1)
    global model
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    


# In[9]:


def train_model() :
    # Compile and train the model
    global x_train, y_train, x_test, y_test, model

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)
    


# In[10]:


def evaluate_model() :
    # Evaluate the model
    global x_train, y_train, x_test, y_test, model

    model.evaluate(x_test, y_test)
    


# In[11]:


def predict_image(image_index) :
    # Predict image 
    global x_train, y_train, x_test, y_test, model

    
    # Validate index must be < 10 000
    if image_index > 10000 :
        image_index = 25
    image = x_test[image_index]
    img_rows, img_cols, i = image.shape

    plt.imshow(image.reshape(28, 28),cmap='Greys')
    pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
    print('The predected image is : ' , pred.argmax())

    # printimage.reshape(28, 28))
    


# In[12]:


def visualaize_train_data() :
    global y_train
    x = y_train.reshape(-1,1)
    df = pd.DataFrame.from_records(x)
    df.columns = [ 'label']
    # print(df['label'].value_counts()) # Frequency distribution
    sns.countplot(df['label'])
    


# In[13]:


# RUN 

# 1 - Load Data
load_data()

# 2 - Preprocess data
pre_process_data()

# 3 - Normalize data
normalize_data()

# 4 - Visualize data
visualaize_train_data()


# 4 - create model
# create_model()

# 5 - train model with x_train, y_train
# train_model()

# 6 - evaluate model
# evaluate_model()


print(x_train.shape)


# In[ ]:


# linear model

model_linear = SVC(kernel='linear')
model_linear.fit(x_train.reshape(60000, 784), y_train)

# predict using x_test
y_pred = model_linear.predict(x_test.reshape(10000, 784))


