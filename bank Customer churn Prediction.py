#!/usr/bin/env python
# coding: utf-8

# ## bank Customer churn Prediction

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.svm          import SVC
from sklearn.pipeline import Pipeline

df = pd.read_csv('Bank_Customer_Churn.csv')
df.head()
df.tail()
df.shape
# drop columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
df.head()
df.isnull()
df.isnull().sum()
df.isnull().mean() * 100
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test =train_test_split(df.drop('Exited',axis=1),df['Exited'], test_size=0.3)

x_train
x_test

# apply onehotencoder
trf1 = ColumnTransformer(transformers=[
    ('ohe_gender',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,2])
],remainder='passthrough')
# apply standerScaler

trf2 = ColumnTransformer(transformers=[
    ('scaler',StandardScaler(),slice(0,14))
],remainder='passthrough')

# apply Algo
#trf3 = RandomForestClassifier(n_estimators=80,min_samples_split=10,max_depth=15,criterion="gini")
trf4 = LogisticRegression()
#trf7 = DecisionTreeClassifier(random_state=0)
#trf8 = SVC(C= .1, kernel='linear', gamma= 1)


# use pipeline

pipe = Pipeline([('trf1',trf1),
                ('trf2',trf2),
                ('trf8',trf8)])
             
               
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

cm
# logistic = 0.8033333333333333 (minmaxscaler)  , StandardScaler = 0.812
# Randomforest = 0.854 (minmaxscaler)           , StandardScaler = 0.8673
# Decisiontree =0.7873(minmaxscaler)            , StandardScaler = 0.8026
# SVM = 0.794666(minmaxscaler

