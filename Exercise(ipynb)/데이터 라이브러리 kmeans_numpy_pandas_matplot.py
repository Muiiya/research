#!/usr/bin/env python
# coding: utf-8

# In[2]:


# K-means 클러스터링 알고리즘 : KMeans 라이브러리 사용

from sklearn.cluster import KMeans


# In[3]:


# 연산 처리를 용이하게 하기 위해 사용

import numpy as np

# 데이터의 포인트를 만들기 위해 사용

import pandas as pd

# 데이터의 시각화를 위해 사용

import seaborn as sb

# matplot 라이브러리 사용

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# 간편히 평면에 표현할 수 있도록 x라는 축과 y라는 축을 가진 데이터 프레임 생성

df = pd.DataFrame(columns=['x', 'y'])


# In[6]:


df.loc[0]=[3, 2]
df.loc[1]=[7, 4]
df.loc[2]=[5, 7]
df.loc[3]=[2, 8]
df.loc[4]=[6, 4]
df.loc[5]=[9, 3]
df.loc[6]=[5, 11]
df.loc[7]=[2, 5]
df.loc[8]=[5, 6]
df.loc[9]=[8, 7]
df.loc[10]=[7, 3]
df.loc[11]=[3, 9]
df.loc[12]=[4, 2]
df.loc[13]=[2, 5]
df.loc[14]=[7, 4]
df.loc[15]=[3, 5]
df.loc[16]=[5, 7]
df.loc[17]=[4, 6]
df.loc[18]=[6, 1]
df.loc[19]=[9, 2]


# In[7]:


# df 데이터 값을 표로 출력

df.head(20)


# In[8]:


# x좌표와 y좌표의 형태로 표현
# df라는 data를 사용
# 데이터 위치 점의 크기는 100으로 설정

sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={"s": 100})
plt.title('! K-means Example !')
plt.xlabel('x_sample')
plt.ylabel('y_sample')


# In[11]:


# K-Means를 활용한 클러스터링을 수행

# df의 값을 numpy의 객체로서 초기화해줌
points =  df.values

# 위 데이터를 기반으로 kmeans 알고리즘을 수행하여 총 클러스터 4개를 생성
kmeans = KMeans(n_clusters=4).fit(points)

# 각 클러스터들의 중심 위치를 구할 수 있도록 세팅
kmeans.cluster_centers_

# 성공적으로 클러스터링이 완료. 결과값으로 총 4개의 클러스터의 중심 위치가 구해진다
# 기본적으로 별다른 명시를 해주지 않으면 자동으로 특정한 위치에서부터 무작위 값을 결정
# 무작위 값을 가지므로 실행할 때마다 클러스터의 중심값은 변한다


# In[12]:


# 어떤 클러스터에 속해있는지 확인
#(3, 1, 4, 2, 1, 1, 2, 3, 4, 4, 1, 2, 3, 3, 1, 3, 4, 4, 1, 1 클러스터에 각각 속한다는)
kmeans.labels_


# In[13]:


# 어떤 클러스터에 속하는지까지 함께 표로 출력

df['cluster']=kmeans.labels_
df.head(20)


# In[14]:


# x좌표와 y좌표 형태로 표현할 것
# df라는 data를 사용
# 데이터 위치 점의 크기는 150으로 설정
# cluster 속성을 기준으로 색깔 구분하여 그래프 출력

sb.lmplot('x', 'y', data=df, fit_reg=False, scatter_kws={"s": 150}, hue="cluster")
plt.title('! K-means Example !')


# In[ ]:




