#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'


# In[25]:


import FinanceDataReader as fdr
samsung = fdr.DataReader('005930')
samsung.tail()


# In[26]:


apple = fdr.DataReader('AAPL')
apple.tail()
apple.head()


# In[27]:


ford = fdr.DataReader('F', '1980-01-01', '2019-12-30')
ford.head()


# In[28]:


ford.tail()


# In[29]:


STOCK_CODE = '005930'


# In[30]:


stock = fdr.DataReader(STOCK_CODE)


# In[31]:


stock.head()


# In[32]:


stock.tail()


# In[33]:


stock.index


# # DatetimeIndex로 정의되어 있다면 연도, 월, 일을 쪼갤 수 있으며, 월별, 연도별 피벗 데이터를 만들 때 유용하게 활용할 수 있다

# In[34]:


stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day


# In[35]:


stock.head()


# In[36]:


plt.figure(figsize=(16,9))
sns.lineplot(y=stock['Close'], x=stock.index)


# In[37]:


time_steps = [['1990', '2000'],
              ['2000', '2010'],
              ['2010', '2015'],
              ['2015', '2020']]
fig, axes = plt.subplots(2,2)
fig.set_size_inches(16,9)
for i in range(4):
    ax = axes[i//2, i%2]
    df = stock.loc[(stock.index > time_steps[i][0])&(stock.index < time_steps[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
plt.show()


# # 데이터 전처리

# In[38]:


from sklearn.preprocessing import MinMaxScaler


# In[39]:


scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled = scaler.fit_transform(stock[scale_cols])
scaled


# In[40]:


df = pd.DataFrame(scaled, columns=scale_cols)


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('Close',1), df['Close'], test_size=0.2, random_state=0, shuffle=False)


# In[43]:


x_train.shape, y_train.shape


# In[44]:


x_test.shape, y_test.shape


# In[45]:


x_train


# In[46]:


import tensorflow as tf


# In[47]:


def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


# In[48]:


WINDOW_SIZE = 20
BATCH_SIZE = 32


# In[49]:


train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)


# In[50]:


for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')


# In[53]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    Conv1D(filters=32, kernel_size=5,
          padding="causal",
          activation="relu",
          input_shape=[WINDOW_SIZE, 1]),
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])


# In[54]:


loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])


# In[55]:


earlystopping = EarlyStopping(monitor='val_loss', patience=10)
filename = os.path.join('tmp', 'checkpointer.ckpt')
checkpoint = ModelCheckpoint(filename,
                            save_weights_only=True,
                            save_best_only=True,
                            monitor='val_loss',
                            verbose=1)


# In[56]:


history = model.fit(train_data,
                   validation_data=(test_data),
                   epochs=50,
                   callbacks=[checkpoint, earlystopping])


# In[57]:


model.load_weights(filename)


# In[58]:


pred = model.predict(test_data)


# In[59]:


pred.shape


# In[60]:


plt.figure(figsize=(12,9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()


# In[ ]:




