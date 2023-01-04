#!/usr/bin/env python
# coding: utf-8

# # Creating NumPy Arrays

# In[ ]:


import numpy as np


# In[ ]:


a1 = np.arange(10)        # creates a range from 0 to 9
print(a1)                 # [0 1 2 3 4 5 6 7 8 9]
print(a1.shape)           # (10,)


# In[ ]:


a2 = np.arange(0,10,2)    # creates a range from 0 to 9, step 2
print(a2)                 # [0 2 4 6 8]


# In[ ]:


a3 = np.zeros(5)          # create an array with all 0s
print(a3)                 # [ 0.  0.  0.  0.  0.]
print(a3.shape)           # (5,)


# In[ ]:


a4 = np.zeros((2,3))      # array of rank 2 with all 0s; 2 rows and 3 columns
print(a4.shape)           # (2,3)
print(a4)


# In[ ]:


a5 = np.full((2,3), 8)    # array of rank 2 with all 8s
print(a5)


# In[ ]:


a6 = np.eye(4)            # 4x4 identity matrix
print(a6)


# In[ ]:


a7 = np.random.random((2,4)) # rank 2 array (2 rows 4 columns) with random values
                             # in the half-open interval [0.0, 1.0)
print(a7)


# In[ ]:


list1 = [1,2,3,4,5]  # list1 is a list in Python
r1 = np.array(list1) # rank 1 array
print(r1)            # [1 2 3 4 5]
print(r1.shape)


# ## Array Indexing

# In[ ]:


print(r1[0])         # 1
print(r1[1])         # 2

print(r1[-1])        # 5
print(r1[-2])        # 4


# In[ ]:


list2 = [6,7,8,9,0]
r2 = np.array([list1,list2])  # rank 2 array
print(r2)
print(r2.shape)               # (2,5) - 2 rows and 5 columns
print(r2[0,0])                # 1
print(r2[0,1])                # 2
print(r2[1,0])                # 6


# In[ ]:


list1 = [1,2,3,4,5]
r1 = np.array(list1)
print(r1[[2,4]])              # [3 5]


# ## Boolean Indexing

# In[ ]:


print(r1>2)     # [False False  True  True  True]


# In[ ]:


print(r1[r1>2])    # [3 4 5]


# # Exercises
# ***
# 

# 1. Print out all the odd number items in the r1 array
# 2. Print out the last third number in the r1 array

# # Solutions
# ---

# In[ ]:


print(r1[r1 % 2 == 1])

print(r1[-3])


# ---
# 

# In[ ]:


nums = np.arange(20)
print(nums)        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]


# In[ ]:


odd_num = nums[nums % 2 == 1]
print(odd_num)     # [ 1  3  5  7  9 11 13 15 17 19]


# ## Slicing Arrays

# In[ ]:


a = np.array([[1,2,3,4,5],
               [4,5,6,7,8],
               [9,8,7,6,5]])    # rank 2 array
print(a)


# In[ ]:


b1 = a[1:3, :3]                 # row 1 to 3 (not inclusive) and first 3 columns
print(b1)


# In[ ]:


b2 = a[-2:,-2:]
print(b2)


# ## NumPy Slice Is a Reference

# In[ ]:


b3 = a[1:, 2:]      # row 1 onwards and column 2 onwards
                    # b3 is now pointing to a subset of a7
print(b3)


# In[ ]:


b3[0,2] = 88         # b3[0,2] is pointing to a[1,4]; modifying it will modify
                     # the original array
print(a)


# In[ ]:


b4 = a[2:, :]        # row 2 onwards and all columns
print(b4)            # b4 is rank 2
print(b4.shape)


# In[ ]:


b5 = a[2, :]         # row 2 and all columns
print(b5)            # b5 is rank 1


# In[ ]:


print(b5.shape)      # (5,)


# # Reshaping Arrays

# In[ ]:


b5 = b5.reshape(1,-1)
print(b5)


# In[ ]:


b4.reshape(-1,)


# # Array Maths

# In[ ]:


x1 = np.array([[1,2,3],[4,5,6]])
y1 = np.array([[7,8,9],[2,3,4]])

print(x1 + y1)


# In[ ]:


x = np.array([2,3])
y = np.array([4,2])
z = x + y


# In[ ]:


np.add(x1,y1)


# In[ ]:


print(x1 - y1)     # same as np.subtract(x1,y1)
print(x1 * y1)     # same as np.multiply(x1,y1)
print(x1 / y1)     # same as np.divide(x1,y1)


# In[ ]:


names   = np.array(['Ann','Joe','Mark'])
heights = np.array([1.5, 1.78, 1.6])
weights = np.array([65, 46, 59])

bmi = weights/heights **2            # calculate the BMI
print(bmi)                           # [ 28.88888889  14.51836889  23.046875  ]


# In[ ]:


print("Overweight: "  , names[bmi>25])                    # Overweight:  ['Ann']
print("Underweight: " , names[bmi<18.5])                  # Underweight:  ['Joe']
print("Healthy: "     , names[(bmi>=18.5) & (bmi<=25)])   # Healthy:  ['Mark']


# # Dot Product

# In[ ]:


x = np.array([2,3])
y = np.array([4,2])
np.dot(x,y)  # 2x4 + 3x2 = 14 


# In[ ]:


x2 = np.array([[1,2,3],[4,5,6]])
y2 = np.array([[7,8],[9,10], [11,12]])
print(np.dot(x2,y2))                     # matrix multiplication


# ## Matrix

# In[ ]:


x2 = np.matrix([[1,2],[4,5]])
y2 = np.matrix([[7,8],[2,3]])


# In[ ]:


x1 = np.array([[1,2],[4,5]])
y1 = np.array([[7,8],[2,3]])
x1 = np.asmatrix(x1)
y1 = np.asmatrix(y1)


# In[ ]:


x1 = np.array([[1,2],[4,5]])
y1 = np.array([[7,8],[2,3]])
print(x1 * y1)     # element-by-element multiplication

x2 = np.matrix([[1,2],[4,5]])
y2 = np.matrix([[7,8],[2,3]])
print(x2 * y2)    # dot product; same as np.dot()


# ## Cumulative Sum

# In[ ]:


a = np.array([(1,2,3),(4,5,6), (7,8,9)])
print(a)


# In[ ]:


print(a.cumsum())   # prints the cumulative sum of all the
                    # elements in the array
                    # [ 1  3  6 10 15 21 28 36 45]


# In[ ]:


print(a.cumsum(axis=0))  # sum over rows for each of the 3 columns


# In[ ]:


print(a.cumsum(axis=1))  # sum over columns for each of the 3 rows


# ## NumPy Sorting

# In[ ]:


ages = np.array([34,12,37,5,13])
sorted_ages = np.sort(ages)   # does not modify the original array
print(sorted_ages)            # [ 5 12 13 34 37]
print(ages)                   # [34 12 37  5 13]


# In[ ]:


ages.sort()                   # modifies the array
print(ages)                   # [ 5 12 13 34 37]


# In[ ]:


ages = np.array([34,12,37,5,13])
print(ages.argsort())         # [3 1 4 0 2]


# In[ ]:


print(ages[ages.argsort()])   # [ 5 12 13 34 37]


# In[ ]:


persons = np.array(['Johnny','Mary','Peter','Will','Joe'])
ages    = np.array([34,12,37,5,13])
heights = np.array([1.76,1.2,1.68,0.5,1.25])


# In[ ]:


sort_indices = np.argsort(ages)  # performs a sort based on ages
                                 # and returns an array of indices
                                 # indicating the sort order


# In[ ]:


print(persons[sort_indices])      # ['Will' 'Mary' 'Joe' 'Johnny' 'Peter']
print(ages[sort_indices])         # [ 5 12 13 34 37]
print(heights[sort_indices])      # [ 0.5   1.2   1.25  1.76  1.68]


# In[ ]:


sort_indices = np.argsort(persons)   # sort based on names
print(persons[sort_indices])         # ['Joe' 'Johnny' 'Mary' 'Peter' 'Will']
print(ages[sort_indices])            # [13 34 12 37  5]
print(heights[sort_indices])         # [ 1.25  1.76  1.2   1.68  0.5 ]


# In[ ]:


reverse_sort_indices = np.argsort(persons)[::-1] # reverse the order of a list
print(persons[reverse_sort_indices])     # ['Will' 'Peter' 'Mary' 'Johnny' 'Joe']
print(ages[reverse_sort_indices])        # [ 5 37 12 34 13]
print(heights[reverse_sort_indices])     # [ 0.5   1.68  1.2   1.76  1.25]


# # Array Assignment

# ## Copying by Reference

# In[ ]:


list1 = [[1,2,3,4], [5,6,7,8]]
a1 = np.array(list1)
print(a1)


# In[ ]:


a2 = a1             # creates a copy by reference
print(a1)
print(a2)


# In[ ]:


a2[0][0] = 11      # make some changes to a2
print(a1)          # affects a1
print(a2)


# In[ ]:


a1.shape = 1,-1   # reshape a1
print(a1)
print(a2)         # a2 also changes shape


# ## Copying by View (Shallow Copy)

# In[ ]:


list1 = [[1,2,3,4], [5,6,7,8]]
a1 = np.array(list1)
a2 = a1.view()    # creates a copy of a1 by reference; but changes
                  # in dimension in a1 will not affect a2
print(a1)
print(a2)


# In[ ]:


a1[0][0] = 11     # make some changes in a1
print(a1)
print(a2)         # changes is also seen in a2


# In[ ]:


a1.shape = 1,-1   # change the shape of a1
print(a1)
print(a2)         # a2 does not change shape


# ## Copying by Value (Deep Copy)

# In[ ]:


list1 = [[1,2,3,4], [5,6,7,8]]
a1 = np.array(list1)
a2 = a1.copy()     # create a copy of a1 by value (deep copy)


# In[ ]:


a1[0][0] = 11     # make some changes in a1
print(a1)
print(a2)         # changes is not seen in a2


# In[ ]:


a1.shape = 1,-1   # change the shape of a1
print(a1)
print(a2)         # a2 does not change shape


# In[ ]:




