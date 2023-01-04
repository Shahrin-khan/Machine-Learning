#!/usr/bin/env python
# coding: utf-8

# # Getting Datasets

# ## Using the Scikit-learn Dataset

# In[1]:


from sklearn import datasets
iris = datasets.load_iris()   # raw data of type Bunch


# In[2]:


print(iris.DESCR)


# In[3]:


print(iris.data)               # Features


# In[4]:


print(iris.feature_names)      # Feature Names


# In[5]:


print(iris.target)             # Labels
print(iris.target_names)       # Label names


# In[6]:


import pandas as pd
df = pd.DataFrame(iris.data)   # convert features
                               # to dataframe in Pandas
print(df.head())


# In[7]:


# data on breast cancer
breast_cancer = datasets.load_breast_cancer()

# data on diabetes
diabetes = datasets.load_diabetes()

# dataset of 1797 8x8 images of hand-written digits
digits = datasets.load_digits()


# # Generating Your Own Dataset

# ## Linearly Distributed Dataset

# In[8]:


 
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=5.4)
plt.scatter(X,y)


# ## Clustered Dataset

# In[9]:


 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(500, centers=3)  # Generate isotropic Gaussian
                                   # blobs for clustering

rgb = np.array(['r', 'g', 'b'])

# plot the blobs using a scatter plot and use color coding
plt.scatter(X[:, 0], X[:, 1], color=rgb[y])


# ## Clustered Dataset Distributed in Circular Fashion

# In[10]:


 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, noise=0.09)

rgb = np.array(['r', 'g', 'b'])
plt.scatter(X[:, 0], X[:, 1], color=rgb[y])


# # Getting Started with Scikit-learn

# In[11]:


 
import matplotlib.pyplot as plt

# represents the heights of a group of people in metres
heights = [[1.6], [1.65], [1.7], [1.73], [1.8]]

# represents the weights of a group of people in kgs
weights = [[60], [65], [72.3], [75], [80]]

plt.title('Weights plotted against heights')
plt.xlabel('Heights in metres')
plt.ylabel('Weights in kilograms')

plt.plot(heights, weights, 'k.')

# axis range for x and y
plt.axis([1.5, 1.85, 50, 90])
plt.grid(True)


# ## Using the LinearRegression Class for Fitting the Model

# In[12]:


from sklearn.linear_model import LinearRegression

# Create and fit the model
model = LinearRegression()
model.fit(X=heights, y=weights)


# ## Making Predictions

# In[13]:


# make prediction
weight = model.predict([[1.75]])[0][0]
print(round(weight,2))         # 76.04


# ## Plotting the Linear Regression Line

# In[14]:


import matplotlib.pyplot as plt

heights = [[1.6], [1.65], [1.7], [1.73], [1.8]]
weights = [[60], [65], [72.3], [75], [80]]

plt.title('Weights plotted against heights')
plt.xlabel('Heights in metres')
plt.ylabel('Weights in kilograms')

plt.plot(heights, weights, 'k.')

plt.axis([1.5, 1.85, 50, 90])
plt.grid(True)

# plot the regression line
plt.plot(heights, model.predict(heights), color='r')


# ## Getting the Gradient and Intercept of the Linear Regression Line

# In[15]:


plt.title('Weights plotted against heights')
plt.xlabel('Heights in metres')
plt.ylabel('Weights in kilograms')

plt.plot(heights, weights, 'k.')

plt.axis([0, 1.85, -200, 200])
plt.grid(True)

# plot the regression line
extreme_heights = [[0], [1.8]]
plt.plot(extreme_heights, model.predict(extreme_heights), color='b')


# In[16]:


round(model.predict([[0]])[0][0],2)   # -104.75


# In[17]:


print(round(model.intercept_[0],2))   # -104.75


# In[18]:


print(round(model.coef_[0][0],2))     # 103.31


# ## Examining the Performance of the Model by Calculating the Residual Sum of Squares

# In[19]:


import numpy as np

print('Residual sum of squares: %.2f' %
       np.sum((weights - model.predict(heights)) ** 2))


# The RSS should be as small as possible, with 0 indicating that the regression line fits the points exactly (rarely achievable in the real world).

# ## Evaluating the Model Using a Test Dataset

# In[20]:


# test data
heights_test = [[1.58], [1.62], [1.69], [1.76], [1.82]]
weights_test = [[58], [63], [72], [73], [85]]


# In[21]:


# Total Sum of Squares (TSS)
weights_test_mean = np.mean(np.ravel(weights_test))
TSS = np.sum((np.ravel(weights_test) -
              weights_test_mean) ** 2)
print("TSS: %.2f" % TSS)

# Residual Sum of Squares (RSS)
RSS = np.sum((np.ravel(weights_test) -
              np.ravel(model.predict(heights_test)))
                 ** 2)
print("RSS: %.2f" % RSS)

# R_squared
R_squared = 1 - (RSS / TSS)
print("R-squared: %.2f" % R_squared)


# In[22]:


# using scikit-learn to calculate r-squared
print('R-squared: %.4f' % model.score(heights_test,
                                      weights_test))

# R-squared: 0.9429


# An R-Squared value of 0.9429 (94.29%) indicates a pretty good fit for your test data.

# ## Persisting the Model

# In[23]:


import pickle

# save the model to disk
filename = 'HeightsAndWeights_model.sav'
# write to the file using write and binary mode
pickle.dump(model, open(filename, 'wb'))


# In[24]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


# In[25]:


result = loaded_model.score(heights_test,
                            weights_test)


# Using the joblib module is very similar to using the pickle module

# In[26]:


from sklearn.externals import joblib

# save the model to disk
filename = 'HeightsAndWeights_model2.sav'
joblib.dump(model, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(heights_test,
                            weights_test)
print(result)


# # Data Cleansing

# ## Cleaning Rows with NaNs

# In[27]:


import pandas as pd
df = pd.read_csv('NaNDataset.csv')
df.isnull().sum()


# In[28]:


print(df)


# ### Replacing NaN with the Mean of the Column

# In[29]:


# replace all the NaNs in column B with the average of column B
df.B = df.B.fillna(df.B.mean())
print(df)


# ### Removing Rows

# In[30]:


df = pd.read_csv('NaNDataset.csv')
df = df.dropna()                             # drop all rows with NaN
print(df)


# In[31]:


df = df.reset_index(drop=True)               # reset the index
print(df)


# ## Removing Duplicate Rows

# In[32]:


import pandas as pd
df = pd.read_csv('DuplicateRows.csv')
print(df.duplicated(keep=False))


# In[33]:


print(df.duplicated(keep="first"))


# In[34]:


print(df[df.duplicated(keep=False)])


# In[35]:


df.drop_duplicates(keep='first', inplace=True)  # remove duplicates and keep the first
print(df)


# In[36]:


df.drop_duplicates(subset=['A', 'C'], keep='last',
                           inplace=True)     # remove all duplicates in
                                             # columns A and C and keep
                                             # the last
print(df)


# ## Normalizing Columns

# In[37]:


import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('NormalizeColumns.csv')
print(df)

x = df.values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=df.columns)
print(df)


# ## Removing Outliers

# ### Tukey Fences

# In[38]:


import numpy as np

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound))


# In[39]:


import pandas as pd
df = pd.read_csv("http://www.mosaic-web.org/go/datasets/galton.csv")
print(df.head())


# In[40]:


print("Outliers using outliers_iqr()")
print("=============================")
for i in outliers_iqr(df.height)[0]:
    print(df[i:i+1])


# ### Z-Score

# In[41]:


def outliers_z_score(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    return np.where(np.abs(z_scores) > threshold)


# In[42]:


print("Outliers using outliers_z_score()")
print("=================================")
for i in outliers_z_score(df.height)[0]:
    print(df[i:i+1])
print()


# In[ ]:




