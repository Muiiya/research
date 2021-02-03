#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 간단한 산수 함수

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[4]:


a = tf.constant(17)
b = tf.constant(5)

sess = tf.Session()


# In[5]:


c = tf.add(a,b)
sess.run(c)


# In[6]:


c = tf.subtract(a,b)
sess.run(c)


# In[7]:


c = tf.multiply(a,b)
sess.run(c)


# In[8]:


# truediv : 나누기

c = tf.truediv(a,b)
sess.run(c)


# In[9]:


# mod : 나머지

c = tf.mod(a,b)
sess.run(c)


# In[10]:


c = -a
sess.run(c)


# In[11]:


# abs : 절대값

c = tf.abs(c)
sess.run(c)


# In[12]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[13]:


a = tf.constant(15.5)
b = tf.constant(3.0)

sess = tf.Session()


# In[14]:


# 음수화 함수

c = tf.negative(a)
sess.run(c)


# In[15]:


# 부호 함수 : 양수일때

c = tf.sign(a)
sess.run(c)


# In[16]:


# 부호 함수 : 음수일때

c = tf.sign(-a)
sess.run(c)


# In[18]:


# 부호 함수 : 0일때

c = tf.sign(0)
sess.run(c)


# In[19]:


#제곱

c = tf.square(b)
sess.run(c)


# In[20]:


# 4제곱

c = tf.pow(b, 4)
sess.run(c)


# In[21]:


# 큰 수를 출력, 여러개 사용 불가

c = tf.maximum(a, b)
sess.run(c)


# In[22]:


# 작은 수를 출력, 여러개 사용 불가

c = tf.minimum(a, b)
sess.run(c)


# In[23]:


# e의 b제곱을 출력

c = tf.exp(b)
sess.run(c)


# In[24]:


# 로그값 출력

c = tf.log(b)
sess.run(c)


# In[25]:


# sin 함수 : b 라디안 > tf.sin(0) = 0

c = tf.sin(b)
sess.run(c)


# In[26]:


# cos 함수 : b 라디안 > tf.cos(0) = 1

c = tf.cos(b)
sess.run(c)


# In[ ]:




