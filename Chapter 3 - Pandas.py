#!/usr/bin/env python
# coding: utf-8

# # Pandas Series

# In[1]:


import pandas as pd
series = pd.Series([1,2,3,4,5])
print(series)


# ## Creating a Series Using a Specified Index

# In[2]:


series = pd.Series([1,2,3,4,5], index=['a','b','c','d','c'])  # note the duplicate index 'c'
print(series)


# ## Accessing Elements in a Series

# In[3]:


print(series[2])          # 3
# same as
print(series.iloc[2])     # 3  - based on the position of the index


# In[4]:


print(series['d'])        # 4
# same as
print(series.loc['d'])    # 4 - based on the label in the index


# In[5]:


print(series['c'])        # more than 1 row has the index 'c'


# In[6]:


print(series[2:])         # returns a Series
print(series.iloc[2:])    # returns a Series


# ## Specifying a Datetime Range as the Index of a Series

# In[7]:


dates1 = pd.date_range('20190525', periods=12)
print(dates1)


# In[8]:


series = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
series.index = dates1
print(series)


# ## Date Ranges

# In[9]:


dates2 = pd.date_range('2019-05-01', periods=12, freq='M')
print(dates2)


# In[10]:


dates2 = pd.date_range('2019-05-01', periods=12, freq='MS')
print(dates2)


# In[11]:


dates2 = pd.date_range('05-01-2019', periods=12, freq='MS')
print(dates2)      


# In[12]:


dates3 = pd.date_range('2019/05/17 09:00:00', periods=8, freq='H')
print(dates3)


# # Pandas DataFrame

# In[13]:


import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),
                  columns=list('ABCD'))
print(df)


# ## Specifying the Index in a DataFrame

# In[14]:


df = pd.read_csv('data.csv')                   # load dataframe from CSV file
days = pd.date_range('20190525', periods=10)
df.index = days
print(df)


# In[15]:


print(df.index)


# In[16]:


print(df.values)


# In[17]:


print(df.describe())


# In[18]:


print(df.mean(0))    # 0 means compute the mean for each columns


# In[19]:


print(df.mean(1))   # 1 means compute the mean for each row


# ## Extracting from DataFrames

# ### Selecting the First and Last Five Rows

# In[20]:


print(df.head())


# In[21]:


print(df.head(8))     # prints out the first 8 rows


# In[22]:


print(df.tail())


# In[23]:


print(df.tail(8))     # prints out the last 8 rows


# ### Selecting a Specific Column in a DataFrame

# In[24]:


print(df['A'])
# same as
print(df.A)


# In[25]:


print(df[['A', 'B']])


# ### Slicing Based on Row Number

# In[26]:


print(df[2:4])


# In[27]:


print(df.iloc[2:4])      # 2 rows


# In[28]:


print(df.iloc[2:5])      # 3 rows


# In[29]:


print(df.iloc[[2,4]])    # 2 rows


# In[30]:


# print(df[[2,4]])   # error; need to use the iloc indexer
print(df.iloc[2])    # prints out row number 2


# ### Slicing Based on Row and Column Numbers

# In[31]:


print(df.iloc[2:4, 1:4])        # 2 rows, 3 columns


# In[32]:


print(df.iloc[[2,4], [1,3]])    # 2 rows, 2 columns


# ### Slicing Based on Labels

# In[33]:


print(df['20190601':'20190603'])


# In[34]:


print(df.loc['20190601':'20190603'])


# In[35]:


print(df.loc['20190601':'20190603', 'A':'C'])


# In[36]:


print(df.loc['20190601':'20190603', ['A','C']])


# In[37]:


print(df.loc['20190601'])


# In[38]:


# print(df.loc[['20190601','20190603']])   # KeyError


# In[39]:


from datetime import datetime
date1 = datetime(2019, 6, 1, 0, 0, 0)
date2 = datetime(2019, 6, 3, 0, 0, 0)
print(df.loc[[date1,date2]])


# In[40]:


print(df.loc[date1, ['A','C']])


# ## Selecting a Single Cell in a DataFrame

# In[41]:


from datetime import datetime
d = datetime(2019, 6, 3, 0, 0, 0)
print(df.at[d,'B'])


# ## Selecting Based on Cell Value

# In[42]:


print(df[(df.A > 0) & (df.B>0)])


# ## Transforming DataFrames

# In[43]:


print(df.transpose())


# In[44]:


print(df.T)


# In[45]:


def checkSeriesOrDataframe(var):
    if isinstance(var, pd.DataFrame):
        return 'Dataframe'
    if isinstance(var, pd.Series):
        return 'Series'


# ## Sorting Data in a DataFrame

# ### Sorting by Index

# In[46]:


print(df.sort_index(axis=0, ascending=False))  # axis = 0 means sort by
                                               # index


# In[47]:


print(df.sort_index(axis=1, ascending=False))  # axis = 1 means sort by
                                               # column


# ### Sorting by Value

# In[48]:


print(df.sort_values('A', axis=0))


# In[49]:


print(df.sort_values('20190601', axis=1))


# ## Applying Functions to a DataFrame

# In[50]:


import math
sq_root = lambda x: math.sqrt(x) if x > 0 else x
sq      = lambda x: x**2


# In[51]:


print(df.B.apply(sq_root))


# In[52]:


print(df.B.apply(sq))


# In[53]:


# df.apply(sq_root)    # ValueError


# In[54]:


df.apply(sq)    


# In[55]:


for column in df:
    df[column] = df[column].apply(sq_root)
print(df)


# In[56]:


print(df.apply(np.sum, axis=0))


# In[57]:


print(df.apply(np.sum, axis=1))


# ## Adding and Removing Rows and Columns in a DataFrame

# In[58]:


import pandas as pd

data = {'name': ['Janet', 'Nad', 'Timothy', 'June', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [6, 13, 14, 1, 7]}

df = pd.DataFrame(data, index =
       ['Singapore', 'China', 'Japan', 'Sweden', 'Norway'])
print(df)


# ### Adding a Column

# In[59]:


import numpy as np

schools = np.array(["Cambridge","Oxford","Oxford","Cambridge","Oxford"])
df["school"] = schools
print(df)


# ### Removing Rows

# In[60]:


print(df.drop(['China', 'Japan']))  # drop rows based on value of index


# In[61]:


print(df[df.name != 'Nad'])         # drop row based on column value


# In[62]:


print(df.drop(df.index[1]))
# same as df.drop['China']


# In[63]:


print(df.drop(df.index[[1,2]]))     # remove the second and third row


# In[64]:


print(df.drop(df.index[-2]))        # remove second last row


# ### Removing Columns

# In[65]:


print(df.drop('reports', axis=1))   # drop column


# In[66]:


print(df.drop(df.columns[1], axis=1))       # drop using columns number


# In[67]:


print(df.drop(df.columns[[1,3]], axis=1))   # drop multiple columns


# ### Generating a Crosstab

# In[68]:


df = pd.DataFrame(
    {
        "Gender": ['Male','Male','Female','Female','Female'],
        "Team"  : [1,2,3,3,1]
    })
print(df)


# In[69]:


print("Displaying the distribution of genders in each team")
print(pd.crosstab(df.Gender, df.Team))


# In[70]:


print(pd.crosstab(df.Team, df.Gender))


# In[ ]:




