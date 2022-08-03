#!/usr/bin/env python
# coding: utf-8

# ## Machine learning Algorithm (Logistic and SVM)

# In[1]:


import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline


# In[2]:


df = pd.read_csv("bank.csv")
df


# In[3]:


df = df[['balance','housing','loan','duration','deposit']]


# In[4]:


df = df.iloc[:,:].values
df


# In[5]:


le = LabelEncoder()


# In[6]:


df[:,4] = le.fit_transform(df[:,4])


# In[7]:


df


# In[8]:


df = pd.DataFrame(df, columns =['balance','housing','loan','duration','deposit'])


# In[9]:


df.head()


# In[10]:


x = df.iloc[:,[0,1,2,3]]
y = df.iloc[:,-1]


# In[11]:


x


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[13]:


trf1 = ColumnTransformer(transformers=[
    ('ohe_gender',OneHotEncoder(sparse=False),[1,2])
],remainder='passthrough')
trf1


# In[14]:


from sklearn.preprocessing import MinMaxScaler

trf2 = ColumnTransformer(transformers=[
    ('scaler',MinMaxScaler(),slice(0,8))
],remainder='passthrough')


# In[15]:


#trf3 = LogisticRegression(solver='lbfgs',penalty='l2')


# In[16]:


trf4 = SVC(C=1.0, kernel='rbf')


# In[17]:


pipe = Pipeline([('trf1',trf1),('trf2',trf2),('trf4',trf4)])


# In[18]:


pipe.fit(x_train,y_train.astype('int'))


# In[19]:


y_pred = pipe.predict(x_test)


# In[20]:


from sklearn.metrics import r2_score
r2_score = (y_test, y_pred)


# In[21]:


r2_score


# In[ ]:




