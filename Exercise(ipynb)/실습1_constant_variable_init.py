#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


#상수 선언

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)
tf.print(c)


# In[10]:


#변수 선언
#Variable(변수)를 생성하여 이용하는 경우에는 초기화가 필요하다

a = tf.Variable(5)
b = tf.Variable(3)
c = tf.multiply(a, b)
init = tf.compat.v1.global_variables_initializer()
tf.print(c)


# In[8]:


a = tf.Variable(15)
tf.print(c)


# In[9]:


a = tf.Variable(15)
c = tf.multiply(a, b)
init = tf.compat.v1.global_variables_initializer()
tf.print(c)


# In[ ]:




