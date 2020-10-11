#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('Salary_Data.csv')
data.head()


# In[5]:


features = data.columns
features = features.drop(['Salary'])
x = data[features]
y = data['Salary']


# In[6]:


plt.scatter (x, y, color = 'red', marker = 'x')
plt.title ('Salary vs YearOfExperience')
plt.xlabel ('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[7]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.25, shuffle=True)


# In[8]:


from sklearn.linear_model import LinearRegression
model = LinearRegression() 


# In[9]:


model.fit (x_train, y_train)


# In[10]:


print ("Slope: ", model.coef_)
print ("Intercept: ", model.intercept_)


# In[11]:


y_pred = model.predict (x_test)
print ("Predicted values of testing data:\n", y_pred)


# In[12]:


from sklearn.metrics import mean_squared_error
cost_function = mean_squared_error (y_test, y_pred)
print("Mean_square_error is: ", cost_function)


# In[13]:


print("Score: ", model.score(x_test,y_test))
#accuracy


# In[14]:


plt.scatter (x_train, y_train, color='blue', marker='x', label='Actual Points')
plt.plot (x_train, model.predict (x_train), color='red', label='Line of Regression')
plt.title ('Salary vs YearOfExperience')
plt.legend()
plt.show()


# In[15]:


plt.scatter (x_test, y_test, color='blue', marker='x', label='Actual Points')
plt.plot(x_test, y_pred, color='red', label='Line of Regression')
plt.legend()
plt.show()
# for testing data


# In[18]:


def salary_predict():
    yr = float(input("Enter years of experience: "))
    salary = model.predict ([[yr]])[0]
    salary = np.round(salary)
    return "Salary of employee: " + str(salary)


# In[23]:


salary_predict()


# In[ ]:




