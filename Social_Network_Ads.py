#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

df = pd.read_csv("Social_Network_Ads.csv")
df.head()
df.shape
df.isnull().sum()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop('Purchased',axis=1),df['Purchased'],test_size=0.3)

# use onehotencoder
trf1 = ColumnTransformer(transformers=[
    ('ohe_gender',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1])
],remainder='passthrough')
trf1
# apply MinMaxscaler
from sklearn.preprocessing import MinMaxScaler

trf2 = ColumnTransformer(transformers=[
    ('scaler',MinMaxScaler(),slice(0,6))
],remainder='passthrough')

trf2
# apply algoritham
trf3 = LogisticRegression()
# use pipeline
pipe = Pipeline([('trf1',trf1),
                ('trf2',trf2),
                ('trf3',trf3)])
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
y_pred
#y_pred = classifier.predict(X_test)
#we build our logistic model and fit it to the training set & we predict our test set result 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr

bias = pipe.score(x_train,y_train)
bias
variance = pipe.score(x_test, y_test)
variance
bias
variance



# In[ ]:




