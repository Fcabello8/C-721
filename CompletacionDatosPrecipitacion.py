#!/usr/bin/env python
# coding: utf-8

# In[1]:


#input precipitation data 
get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import numpy as np
TodasEstaciones = pd.read_csv('C:/Users/FRANCIS CABELLO/Desktop/Final_Project/Scripts/Estacion.csv',index_col=0,parse_dates=True)
TodasEstaciones.head()


# In[2]:


TodasEstaciones.loc['Ene-64':'Dic-11'].plot(subplots=True, figsize=(12, 8))
plt.legend(loc='best')
xticks(rotation='vertical')


# In[3]:


#creation of a correlation map
import seaborn as sns
corr = TodasEstaciones.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# # Training data between Ene-93:Dic-11

# In[4]:


#definition of training sets
X_train = TodasEstaciones.loc['Ene-93':'Dic-11',['Est1','Est3']].astype(float32).values #Est 1, 3
y_train = TodasEstaciones.loc['Ene-93':'Dic-11','Est2'].astype(float32).values #Est 2


# In[5]:


#data normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)


# In[6]:


from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(12, activation='linear', input_shape=(2,)))
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[7]:


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=200,verbose=0)


# In[8]:


y_pred = model.predict(X_train)


# In[10]:


plot(TodasEstaciones.loc['Ene-93':'Dic-11'].index,y_pred,label='Predicted')
TodasEstaciones['Est2'].loc['Ene-93':'Dic-11'].plot()
figsize(12,8)
ylim(0,350)
legend(loc='best')


# # Predict missing data between Ene-81:Dic-92

# In[11]:


#get the prediction for the train set
X_missing = TodasEstaciones.loc['Ene-81':'Dic-92',['Est1','Est3']].astype(float32).values


# In[12]:


#data normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_missing)

X_missing = scaler.transform(X_missing)


# In[13]:


y_missing = model.predict(X_missing)
y_missing = y_missing.reshape([144]).tolist()


# In[14]:


TodasEstaciones['Est2_Completed']=TodasEstaciones['Est2']
TodasEstaciones['Est2_Completed'].loc['Ene-81':'Dic-92']=y_missing


# In[16]:


TodasEstaciones.loc['Ene-81':'Dic-11',['Est1','Est2','Est2_Completed','Est3']].plot(subplots=True, figsize=(15, 10))
plt.legend(loc='best')
xticks(rotation='vertical')
ylim(0,300)


# In[ ]:




