#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:


import os
for dirname, _, filenames in os.walk(r'C:\Users\Aditri\OneDrive'):
    for filename in filenames:
        if filename == 'creditcard.csv':
            print(os.path.join(dirname, filename))


# In[32]:


credit_card=pd.read_csv(r"C:\Users\Aditri\OneDrive\Desktop\creditcard.csv")
credit_card


# In[33]:


credit_card.head()


# In[34]:


credit_card.tail()


# In[35]:


credit_card.info()


# In[36]:


credit_card.describe()


# In[47]:


# checking the number of missing values in each column
credit_card.isnull().sum()


# In[37]:


credit_card['Class'].value_counts()


# Dataset is clearly unblanced 

# In[49]:


#pre-processing
fraud=credit_card[credit_card["Class"]==1]
true=credit_card[credit_card["Class"]==0]
print(fraud)
print(true)
print(fraud.shape)
print(true.shape)


# In[43]:


fraud.Amount.describe()


# In[44]:


true.Amount.describe()


# In[51]:


# compare the values for both transactions
credit_card.groupby('Class').mean()


# In[52]:


true_sample=no_fraud.sample(n=492)
true_sample


# In[46]:


#normalize and standardize the dataset
balanced_dataset=pd.concat([fraud,true],axis=0)
balanced_dataset


# In[54]:


balanced_dataset.head()


# In[53]:


balanced_dataset.tail()


# In[25]:


balanced_dataset["Class"].value_counts()


# In[26]:


x=balanced_dataset.drop("Class",axis='columns')
y=balanced_dataset["Class"]


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[55]:


print(x.shape, x_train.shape, x_test.shape)


# In[56]:


logisticmodel=LogisticRegression()
logisticmodel.fit(x_train,y_train)


# In[58]:


y_train_predict=logisticmodel.predict(x_train)


# In[59]:


accuracy=accuracy_score(y_train,y_train_predict)
print("Accuracy= ",int(accuracy*100),'%')


# In[57]:


y_predict=logisticmodel.predict(x_test)


# In[60]:


accuracy=accuracy_score(y_test,y_predict)
print("Accuracy= ",int(accuracy*100),'%')


# In[ ]:




