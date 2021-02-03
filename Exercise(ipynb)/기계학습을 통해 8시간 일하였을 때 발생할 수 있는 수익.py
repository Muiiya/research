#!/usr/bin/env python
# coding: utf-8

# In[11]:


#텐서플로우 import

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[12]:


# xData, yData 배열 선언

xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]


# In[13]:


# -100부터 100까지의 수 중 임의의 수를 W, b의 값으로 선언(weight/bios)

W = tf.Variable(tf.random.uniform([1], -100, 100))
b = tf.Variable(tf.random.uniform([1], -100, 100))


# In[14]:


#X와 y를 placeholder라는 틀을 만들어 설정 : 가장 대표적인 형태임(앞으로 많이 보게 됨)
#tf.compat.v1.placeholder가 오류 생기는 경우 import tensorflow as tf를 
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()로 바꿔준다

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[16]:


#가설 식을 모델로 제시해줌 - 이 예제에서는 일차방정식 형태로 모델을 제시해봄
H = W*X+b

#비용함수를 정의
cost = tf.reduce_mean(tf.square(H-Y)) # (H-Y)제곱값의 평균값


# In[17]:


#경사 하강 알고리즘에서 얼마만큼 점프할지에 대해 정의해줌
#변동폭을 0.01로 설정 / 너무 커도, 너무 적어도 안되는 값으로, 해당 값 설정이 굉장히 중요
a = tf.Variable(0.01)


# In[18]:


#경사 하강 법으로 최적값을 찾겠다는 설정
optimizer = tf.compat.v1.train.GradientDescentOptimizer(a)


# In[19]:


#cost를 최소화하는 최적값을 train에 설정
train = optimizer.minimize(cost)


# In[20]:


#변수 초기화
init = tf.compat.v1.global_variables_initializer()


# In[21]:


# tensorflow 객체에서 세션을 얻어와 sess라는 변수에 삽입
sess = tf.compat.v1.Session()


# In[22]:


# 세션 초기화
sess.run(init)


# In[23]:


# 학습 진행
for i in range(5001):
    sess.run(train, feed_dict={X: xData, Y: yData})  # x,y 데이터 매칭
    if i % 500 == 0:  #단순히 과정을 보기 위해 500번 간격으로 출력되도록 세팅, 진행과정의 일부를 볼 수 있게 해줌
        print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))


# In[24]:


#최종결과 출력

print(sess.run(H, feed_dict={X: [8]}))  # 최종 결과 모델인 H에 기존에 없었던 8이라는 노동시간을 넣어주었을 때의 결과값 출력 


# In[ ]:




