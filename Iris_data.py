#!/usr/bin/env python
# coding: utf-8

# In[163]:


#Importing the required libraries
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import pickle


# In[164]:


#Data understanding
#Reading the dataset
df = pd.read_csv("Iris.csv")
# Show first five rows from data set
df.head()


# In[165]:


#Checking the shape
df.shape


# In[166]:


#count each species
df['Species'].value_counts()


# In[167]:


#Checking the Metadata Information
df.info()


# In[168]:


#Checking how data is spread
df.describe()


# In[169]:


#Checking correlation
plt.figure(figsize=(6,6))
corrmat = df.drop('Id',axis=1).corr()
sns.heatmap(corrmat, annot=True, square= True,cmap='rainbow')
plt.show()


# In[170]:


#Droping the Id and Species columns
df.drop(['Id', 'Species'],axis='columns',inplace=True)


# In[171]:


df.head()


# In[172]:


#Visualizing the data
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.scatter(df['SepalLengthCm'],df['SepalWidthCm'],color='blue')
plt.xlabel('SepalLength(Cm)')
plt.ylabel('SepalWidth(Cm)')

plt.subplot(1,2,2)
plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],color='blue')
plt.xlabel('PetalLength(Cm)')
plt.ylabel('PetalWidth(Cm)')


# In[173]:


#Taking only the petallength and Petalwidth columns
X=df.iloc[:,2:]
X.head()


# In[174]:


#Droping the Sepallength and Sepalwidth columns
df.drop(['SepalLengthCm','SepalWidthCm'],axis='columns',inplace=True)
df.head()

# # DBSCAN Clustering

# In[202]:


#Importing libraries for DBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics


# In[203]:


#Model training
db= DBSCAN(eps=0.3,min_samples=5)
db_predict=db.fit_predict(df)
db_predict


# In[204]:


df['dbscan_predicted'] = db_predict
df.head()


# In[205]:


df['dbscan_predicted'].value_counts()


# In[206]:


plt.scatter(df['PetalLengthCm'],df['PetalWidthCm'],c=db_predict,cmap='Paired')


# In[207]:


score_dbsacn_s = silhouette_score(df, db_predict)
print('Silhouette Score: %.4f' % score_dbsacn_s)


pickle.dump(db,open("Iris_data.pkl","wb"))