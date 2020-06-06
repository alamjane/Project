#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Downloads\cardio_train.csv", sep=";")

df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


from matplotlib import rcParams
rcParams['figure.figsize'] = 9, 6
df['years'] = (df['age'] / 365).round().astype('int')
sns.countplot(x='years', hue='cardio', data = df, palette="Set2");


# ### It can be observed that people over 55 of age are more exposed to CVD. 
# ### From the table above, we can see that there are outliers in ap_hi, ap_lo, weight and height. We will deal with them later.

# #### Let's look at categorical variables in the dataset and their distribution:

# In[6]:


df_categorical = df.loc[:,['cholesterol','gluc', 'smoke', 'alco', 'active']]
sns.countplot(x="variable", hue="value",data= pd.melt(df_categorical))


# #### Bivariate analysis
# #### It may be useful to split categorical variables by target class:

# In[7]:


df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio",
                data=df_long, kind="count");


# #### It can be clearly seen that patients with CVD have higher cholesterol and blood glucose level. And, generally speaking less active.
# 
# #### To figure out whether "1" stands for women or men in gender column, let's calculate the mean of height per gender. We assume that men are higher than women on average.

# In[8]:


df.groupby('gender')['height'].mean()


# #### Average height for "2" gender is greater, than for "1" gender, therefore "1" stands for women. Let's see how many men and women presented in the dataset:
# 

# In[9]:


df['gender'].value_counts()


# #### Who more often report consuming alcohol - men or women?

# In[10]:


df.groupby('gender')['alco'].sum()


# #### So, men consume alcohol more frequently on average. Next, the target variables are balanced:

# In[11]:


df['cardio'].value_counts(normalize=True)


# In[12]:


# Cross Tab

pd.crosstab(df['cardio'],df['gender'],normalize=True)


# #### Cleaning Data

# In[13]:


df.isnull().values.sum()


# #### Let's remove weights and heights, that fall below 2.5% or above 97.5% of a given range.
# 

# In[14]:


df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,inplace=True)


# #### Let's get rid of the outliers, moreover blood pressure could not be negative value!

# In[15]:


df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)


# In[16]:


blood_pressure = df.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))


# #### Let's create a new feature - Body Mass Index (BMI):
# 
# #### BMI = weight (kg) / [height (m)]2
# 
# #### and compare average BMI for healthy people to average BMI of ill people. Normal BMI values are said to be from 18.5 to 25.

# In[17]:


df['BMI'] = df['weight']/((df['height']/100)**2)
sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=df, color = "yellow",kind="box", height=10, aspect=.7);


# #### Drinking women have higher risks for CVD than drinking men based on thier BMI

# In[18]:


# Correlation Matrix:

plt.figure(figsize=(12,6)) 
sns.heatmap(df.corr(), annot=True)


# #### Splitting the dataset to Train and Test

# In[19]:


x = df.drop(['cardio' ], axis=1)


# In[20]:


x.head()


# In[21]:


y = df['cardio']


# In[22]:


from sklearn import model_selection


# In[24]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0) #80/20 split


# In[25]:


x_train.shape, y_train.shape


# In[26]:


x_test.shape, y_test.shape


# In[27]:


import warnings
warnings.filterwarnings('ignore')


# #### Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()


# In[29]:


model.fit(x_train, y_train)


# In[30]:


prediction=model.predict(x_test)
prediction


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)


# In[44]:


from sklearn.metrics import confusion_matrix

matrix= confusion_matrix(y_test, prediction)

sns.heatmap(matrix,annot = True, fmt = "d")


# In[42]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

print(confusion_matrix(y_test,prediction))


# In[46]:


print(classification_report(y_test,prediction))


# #### SVM

# In[33]:


from sklearn.svm import SVC
clf = SVC()


# In[34]:


clf.fit(x_train, y_train)


# In[35]:


pred = clf.predict(x_test)
pred


# In[36]:


accuracy_score(y_test, pred)


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

print(confusion_matrix(y_test,pred))


# In[47]:


print(classification_report(y_test,pred))


# #### Random Forest

# In[37]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[38]:


rfc.fit(x_train, y_train)


# In[39]:


predict = rfc.predict(x_test)

predict


# In[40]:


accuracy_score(y_test, predict)


# In[48]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

print(confusion_matrix(y_test,predict))


# In[49]:


print(classification_report(y_test,predict))


# In[ ]:




