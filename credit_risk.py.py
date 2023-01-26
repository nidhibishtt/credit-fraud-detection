#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import warnings
import os
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np 


# In[2]:


DeprecationWarning('ignore')
#importing dataset from the files
os.chdir('C:/Users/HP/Desktop/machine_learning')
dataset = pd.read_csv("credit_risk.csv",)


# In[3]:


plt.hist(dataset.ApplicantIncome)
lower_bound=0.1
upper_bound=0.95
res=dataset['ApplicantIncome'].quantile([lower_bound,upper_bound])
print(res)


# In[4]:


true_index=(res.loc[lower_bound]<dataset.ApplicantIncome.values) &     (dataset.ApplicantIncome.values < res.loc[upper_bound])
print(true_index)
dataset.ApplicantIncome=dataset.ApplicantIncome[true_index]


# In[5]:


plt.hist(dataset.CoapplicantIncome)
lower_bound=0.1
upper_bound=0.95
res=dataset['CoapplicantIncome'].quantile([lower_bound,upper_bound])
print(res)


# In[6]:


true_index=(res.loc[lower_bound]<dataset.CoapplicantIncome.values) &     (dataset.CoapplicantIncome.values < res.loc[upper_bound])
print(true_index)
dataset.CoapplicantIncome=dataset.CoapplicantIncome[true_index]


# In[7]:


plt.hist(dataset.LoanAmount)
lower_bound=0.1
upper_bound=0.95
res=dataset['LoanAmount'].quantile([lower_bound,upper_bound])
print(res)


# In[8]:


true_index=(res.loc[lower_bound]<dataset.LoanAmount.values) &     (dataset.LoanAmount.values < res.loc[upper_bound])
print(true_index)
dataset.LoanAmount=dataset.LoanAmount[true_index]


# In[9]:


dataset=dataset.drop('Loan_ID',axis=1)
dataset=dataset.drop('Loan_Amount_Term',axis=1)
dataset=dataset.drop('Married',axis=1)


# In[10]:


dataset['Dependents']=dataset['Dependents'].str.replace('+','').astype(float)  


# In[11]:


dataset['Gender'].fillna('Male',inplace=True)
dataset['Dependents'].fillna(0,inplace=True)
dataset['Self_Employed'].fillna('No',inplace=True)
dataset['LoanAmount'].fillna(146.4121,inplace=True)
dataset['Credit_History'].fillna(1,inplace=True)
dataset['CoapplicantIncome'].fillna(2251.53,inplace=True)
dataset['ApplicantIncome'].fillna(3909.5,inplace=True)


# In[12]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
dataset['Gender']=label.fit_transform(dataset['Gender'])
dataset['Education']=label.fit_transform(dataset['Education'])
dataset['Self_Employed']=label.fit_transform(dataset['Self_Employed'])
dataset['Property_Area']=label.fit_transform(dataset['Property_Area'])
dataset['Loan_Status']=label.fit_transform(dataset['Loan_Status'])


# In[13]:


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,9].values


# In[59]:


X_train, X_test , Y_train , Y_test = train_test_split(X,Y, test_size = 0.2 ,random_state = 1 )


# In[60]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


# In[62]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=3)
clf.fit(X_train,Y_train)
predi=clf.predict(X_test)
score1= accuracy_score(Y_test, predi)
print(score1)


# In[63]:


from sklearn.model_selection import cross_val_score
CV=cross_val_score(clf,X_train,Y_train,cv=5)
print(CV.mean())


# In[ ]:




