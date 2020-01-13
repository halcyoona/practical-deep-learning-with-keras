#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Recognizing Digits
from sklearn import datasets


# In[4]:


digits = datasets.load_digits()


# In[5]:


print(digits.data)


# In[6]:


print(digits.data[0])


# In[7]:


print(digits.data[0].shape)


# In[8]:


print(digits.target[0])


# In[9]:


print(digits.target[-10:])


# In[10]:


print(digits.target.shape)


# In[11]:


print(digits.data.shape)


# In[ ]:


#Learn to predict the class

