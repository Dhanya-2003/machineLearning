#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('House_Price.csv')
data.head()


# In[3]:


x = data.iloc[:,1:7].values
y = data.iloc[:,-1].values


# In[4]:


y = y.reshape(len(y),1)


# In[5]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20,random_state = 0)


# In[6]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[7]:


model.fit(xtrain,ytrain)


# In[8]:


y_pred = model.predict(xtest)


# In[12]:


from sklearn.metrics import mean_squared_error
Cost_function = mean_squared_error(ytest,y_pred)
weights = model.coef_
Intercept = model.intercept_


# In[13]:


import pickle
filename = 'Real estate.csv'
pickle.dump(model,open(filename,'wb'))


# In[14]:


my_model = pickle.load(open(filename,'rb'))


# In[15]:


def House_Price_predict():
    X1 = float(input("Transaction date : "))
    X2 = float(input("House age : "))
    X3 = float(input("Distance to nearest MRT station : "))
    X4 = int(input("No of conveninece stores : "))
    X5 = float(input("Latitude : "))
    X6 = float(input("Longitude : "))
    price = my_model.predict([[X1,X2,X3,X4,X5,X6]])[0][0]
    return "House Price of unit area : " + str(price)


# In[16]:


House_Price_predict()


# In[ ]:




