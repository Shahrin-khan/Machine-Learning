#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# ## Using the Boston Dataset

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
dataset = load_boston()


# In[2]:


print(dataset.feature_names)


# In[3]:


print(dataset.DESCR)


# In[4]:


print(dataset.target)


# In[5]:


df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head()


# In[6]:


df['MEDV'] = dataset.target
df.head()


# ## Data Cleansing

# In[7]:


df.info()


# In[8]:


print(df.isnull().sum())


# ## Feature Selection

# In[9]:


corr = df.corr()
print(corr)


# In[10]:


#---get the top 3 features that has the highest correlation---
print(df.corr().abs().nlargest(3, 'MEDV').index)

#---print the top 3 correlation values---
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])


# ## Multiple Regression

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')


# In[12]:


plt.scatter(df['RM'], df['MEDV'], marker='o')
plt.xlabel('RM')
plt.ylabel('MEDV')


# In[13]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['LSTAT'],
           df['RM'],
           df['MEDV'],
           c='b')

ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
plt.show()


# ## Training the Model

# In[14]:


x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3,
                                                    random_state=5)


# In[16]:


print(x_train.shape)
print(Y_train.shape)


# In[17]:


print(x_test.shape)
print(Y_test.shape)


# In[18]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, Y_train)


# In[19]:


price_pred = model.predict(x_test)


# In[20]:


print('R-squared: %.4f' % model.score(x_test,
                                      Y_test))


# In[21]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, price_pred)
print(mse)

plt.scatter(Y_test, price_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")


# ## Getting the Intercept and Coefficients

# In[22]:


print(model.intercept_)
print(model.coef_)


# In[23]:


print(model.predict([[30,5]]))


# ## Plotting the 3D Hyperplane

# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_boston
dataset = load_boston()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target

x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x['LSTAT'],
           x['RM'],
           Y,
           c='b')

ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")

#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1)   #---for LSTAT---
y_surf = np.arange(0, 10, 1)   #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, Y)

#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ + model.coef_[0] * x + model.coef_[1] * y)

ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

plt.show()


# # Exercises
# ---

# 1. Try the above in a python file. 

# # Polynomial Regression

# In[25]:


df = pd.read_csv('polynomial.csv')
plt.scatter(df.x,df.y)


# In[26]:


model = LinearRegression()

x = df.x[0:6, np.newaxis]     #---convert to 2D array---
y = df.y[0:6, np.newaxis]     #---convert to 2D array---

model.fit(x,y)

#---perform prediction---
y_pred = model.predict(x)

#---plot the training points---
plt.scatter(x, y, s=10, color='b')

#---plot the straight line---
plt.plot(x, y_pred, color='r')
plt.show()

#---calculate R-squared---
print('R-squared for training set: %.4f' % model.score(x,y))


# ## Formula for Polynomial Regression

# A polynomial function of degree 1 has the following form:
# 
# Y = β0 + β1x
# 
# Quadratic regression is a degree 2 
# 
# Y = β0 + β1x + β2x2
# 
# For a polynomial of degree 3, the formula is as follows:
# 
# Y = β0 + β1x + β2x2 + β3x3
# 
# In general, a polynomial of degree n has the formula of:
# 
# Y = β0 + β1x + β2x2 + β3x3 + … + βnxn
# 
# The idea behind polynomial regression is simple — find the coefficients of the polynomial function that best fits the data.
# 

# ## Polynomial Regression in Scikit-learn

# In[27]:


from sklearn.preprocessing import PolynomialFeatures
degree = 2

polynomial_features = PolynomialFeatures(degree = degree)


# In[28]:


x_poly = polynomial_features.fit_transform(x)
print(x_poly)


# In[29]:


model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

#---plot the points---
plt.scatter(x, y, s=10)

#---plot the regression line---
plt.plot(x, y_poly_pred)
plt.show()


# In[30]:


print(model.intercept_)
print(model.coef_)


# In[31]:


print('R-squared for training set: %.4f' % model.score(x_poly,y))


# ## Using Polynomial Multiple Regression on the Boston Dataset

# In[32]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

dataset = load_boston()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target

x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']

from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3,
                                                    random_state=5)


# In[33]:


#---use a polynomial function of degree 2---
degree = 2
polynomial_features= PolynomialFeatures(degree = degree)
x_train_poly = polynomial_features.fit_transform(x_train)


# When using a polynomial function of degree 2 on two independent variables x1 and x2, the formula becomes:
# 
# Y = β0 + β1x1 + β2x2 + β3x12 + β4x1x2 +β5x22
# 
# where Y is the dependent variable, β0 is the intercept, β1, β2, β3, and β4 are the coefficients of the various combinations of the two features x1 and x2, respectively.
# 

# In[34]:


#---print out the formula---
print(polynomial_features.get_feature_names(['x','y']))


# In[35]:


model = LinearRegression()
model.fit(x_train_poly, Y_train)


# In[36]:


x_test_poly = polynomial_features.fit_transform(x_test)
print('R-squared: %.4f' % model.score(x_test_poly,
                                      Y_test))


# In[37]:


print(model.intercept_)
print(model.coef_)


# With these values, the formula now becomes:
# 
# Y = β0 + β1x1 + β2x2 + β3x12 + β4x1x2 +β5x22
# 
# Y = 26.9334305238 + 1.47424550e+00 x1 + (-6.70204730e+00) x2 + 7.93570743e-04 x12 + (-3.66578385e-01) x1x2 + 1.17188007e+00 x22
# 

# ## Plotting the 3D Hyperplane

# In[38]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

dataset = load_boston()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target

x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x['LSTAT'],
           x['RM'],
           Y,
           c='b')

ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")

#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1)   #---for LSTAT---
y_surf = np.arange(0, 10, 1)   #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

#---use a polynomial function of degree 2---
degree = 2
polynomial_features= PolynomialFeatures(degree = degree)
x_poly = polynomial_features.fit_transform(x)
print(polynomial_features.get_feature_names(['x','y']))

#---apply linear regression---
model = LinearRegression()
model.fit(x_poly, Y)

#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ +
                (model.coef_[1] * x) +
                (model.coef_[2] * y) +
                (model.coef_[3] * x**2) +
                (model.coef_[4] * x*y) +
                (model.coef_[5] * y**2))

ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
                rstride=1,
                cstride=1,
                color='None',
                alpha = 0.4)

plt.show()


# # Exercises
# ---

# 1. Try the above in a python file.

# In[ ]:




