#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv("student_info.csv")
df.head()


# In[9]:


df.info


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[36]:


df = df.fillna(df.mean())


# In[37]:


df.head()


# In[38]:


plt.scatter(x=df.study_hours, y=df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()


# In[39]:


x = df.drop("student_marks", axis = "columns")
y = df.drop("study_hours", axis = "columns")
print("shape of X = ", X.shape)
print("shape of y = ", y.shape)


# In[40]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3, random_state=51)


# In[41]:


from sklearn.linear_model import LinearRegression
    
lr = LinearRegression()


# In[42]:


lr.fit(x_train,y_train)


# In[43]:


lr.coef_


# In[44]:


lr.intercept_


# In[45]:


m = 3.93
c = 50.44
y  = m * 4 + c 
y


# In[46]:


y_pred = lr.predict(x_test)
y_pred


# In[48]:


pd.DataFrame(np.c_[x_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])


# In[51]:



lr.score(x_test,y_test)


# # save model

# In[53]:


import joblib
joblib.dump(lr, "student_mark_predictor.pkl2")


# In[54]:


model = joblib.load("student_mark_predictor.pkl")


# In[55]:


model.predict([[5]])[0][0]


# In[56]:


model.predict([[6]])[0][0]


# In[ ]:




