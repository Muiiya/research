#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 플레이스 홀더 : 학습 데이터를 포함하는 변수
# placeholder를 이용하여 입력값과 설계된 수식을 완전히 분리함으로서 보다 간단하게
# 어떠한 데이터를 통해 기계학습을 시키고 관리할 수 있다


# 여러 개의 데이터에 대하여 각각 5와 더한 값을 반환하는 프로그램을 
# 만들어보라는 미션에 대한 솔루션을 만들어보라

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

input = [1, 2, 3, 4, 5]
x = tf.placeholder(dtype = tf.float32)
y = x + 5
sess = tf.Session()
sess.run(y, feed_dict={x: input})


# In[3]:


# 수학 점수와 영어 점수의 평균을 구하는 코드를 짜보자

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

mathscore = [80, 82, 84, 86, 88]
englishscore = [70, 73, 76, 79, 82]

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
y = (a + b) / 2

sess = tf.Session()
sess.run(y, feed_dict={a: mathscore, b: englishscore})


# In[ ]:




