
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import tensorflow as tf
from sklearn import datasets

import matplotlib.pyplot as plt


# In[2]:

digits = datasets.load_digits()
plt.imshow(digits.images[0], interpolation='nearest', cmap='Greys')


# In[14]:

tf.reset_default_graph()

sess = tf.InteractiveSession()#build computational graph

digits.images = digits.images.reshape((-1, 8*8))

inputs = tf.placeholder(tf.float32, digits.images.shape)
W = tf.Variable(tf.truncated_normal([8*8,16]))
mid = tf.matmul(inputs,W)
recon = tf.matmul(mid,tf.transpose(W))

#set up loss and optimiser
loss = tf.reduce_mean((inputs - recon)**2)
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

sess.run(tf.initialize_all_variables())

#train
for i in range(10000):
    L,_ = sess.run([loss,train_step],feed_dict = {inputs:digits.images})
    print('\r{}'.format(L), end='')

dim_red = sess.run(mid,feed_dict = {inputs:digits.images})
plt.scatter(dim_red[:,0],dim_red[:,1])


# In[15]:

plt.imshow(sess.run(recon, feed_dict={inputs:digits.images})[0].reshape((8,8)), 
           interpolation='nearest',
           cmap='Greys')


# In[ ]:



