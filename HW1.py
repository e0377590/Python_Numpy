#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# ### Numpy

# Task1

# In[ ]:



a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])


# In[15]:


a


# In[16]:


mean_a = np.mean(a, axis = 0)


# In[12]:


mean_a


# Task2

# In[18]:


a_centered = np.subtract(a, mean_a)


# In[19]:


a_centered


# Task3

# In[34]:


v1 = a_centered[:, :1]
v1 = v1.transpose()


# In[35]:


v2 = a_centered[:, 1:]


# In[36]:


v2


# In[37]:


a_centered_sp = np.dot(v1,v2)


# In[38]:


a_centered_sp


# In[40]:


N = a.shape


# In[43]:


N[0]


# In[53]:


cov = a_centered_sp/(N[0]-1)


# In[54]:


cov


# Task4

# In[55]:


c = np.cov(a.transpose())


# In[56]:


c[0,1]


# ### Pandas

# Task1

# In[58]:


import pandas as pd


# In[59]:


a = {
    "author_id": [1, 2, 3],
    "author_name": ['Тургенев', 'Чехов', 'Островский']
}

author = pd.DataFrame(a)

author


# In[62]:


b = {
    "author_id": [1, 1, 1, 2, 2, 3, 3],
    "book_title": ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    "price": [450, 300, 350, 500, 450, 370, 290]
}

book = pd.DataFrame(b)

book


# Task2

# In[66]:


authors_price = pd.merge(author, book, on='author_id', how='inner')


# In[67]:


authors_price


# Task3

# In[68]:


top5 = authors_price.nlargest(5, "price")


# In[69]:


top5


# Task4

# In[70]:


df1 = authors_price.groupby('author_name').agg({'price': 'min'}).rename(columns={'price':'min_price'})


# In[71]:


df1


# In[72]:


df2 = authors_price.groupby('author_name').agg({'price': 'max'}).rename(columns={'price':'max_price'})


# In[73]:


df2


# In[76]:


df3 = authors_price.groupby('author_name').agg({'price': 'mean'}).rename(columns={'price':'mean_price'})


# In[77]:


df3


# In[78]:


authors_stat=pd.concat([df1, df2, df3], axis = 1)

authors_stat


# In[ ]:




