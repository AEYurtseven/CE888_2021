#!/usr/bin/env python
# coding: utf-8

# In[1]:


##This is the python file used to do some data analysis and test some pre-processing techniques.
##The data exploration has been done by using the code from 
##https://www.kaggle.com/bkamphaus/exploratory-image-analysis

##import the library 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##This will be used for image manipulation
import sklearn
##glob and os libraries will be used to get data from file system
import glob, os


# In[2]:


#Load a file from file system file path
#saying No_Fire indicate this file should be labelled as no_fire
first = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/lake_resized_lake_frame0.jpg")
#Get the shape of the picture
dims = np.shape(first)
#Picture has 254 columns and 254 rows and 3 different colours
print(dims)


# In[3]:


pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
##Shows the all number of the pixel
print(np.shape(pixel_matrix))


# In[4]:


##show a picture loaded in line 1
##According to picture, fire can be seen but by using pre-processing, 
##there will be potential differences can be observed 
plt.imshow(first)


# In[5]:


fifth = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/lake_resized_lake_frame5.jpg")
dims = np.shape(fifth)

dims


# In[6]:


plt.imshow(fifth)


# In[7]:


#Load a file from file system file path
#saying No_Fire indicate this file should be labelled as fire/
#there is an observable difference between this picture and picture before which can also 
#observable by using pre-processing
first_fire = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame0.jpg")
dims = np.shape(first_fire)

#this shape seems to be same with all of the picture possibly 
#as a consequence of the pre-processing already done. This manipulation done before
#the beginning of this project
print(dims)
plt.imshow(first_fire)


# In[8]:


pixel_fire_matrix = np.reshape(first_fire, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_fire_matrix))


# In[9]:


# sklean is an library used for using macine learning
from sklearn import cluster
##kmeans is a clustering algoritm for unsupervised learning 
##clustering algorithms can be used  for ata exploration 
##For tabular dataset for example, we can gather different data according to their
##similarities to understand the trends in the data, however we will use images as data for this 
##project

##5 centeroids will be used
kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

##Clustering could be useful by clustering  the images. It could be useful as seen
##examples below

##Example 1
##The data is the clustered version of the picture in line 1
dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)


# In[10]:


##It is show the image below is clustered version of line 7

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_fire_matrix)

dims = np.shape(first_fire)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)


# In[11]:


##Image below taken from Fire folder which is a 

TwoThSMt_fire = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame2592.jpg")
dims = np.shape(TwoThSMt_fire)
print(dims)
plt.imshow(TwoThSMt_fire)


# In[12]:


pixel_fire_matrix = np.reshape(TwoThSMt_fire, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_fire_matrix))


# In[13]:


##Here, clustering has been used. Picture shown is the clustered version 
##of the picture in line 11. The image below seems to be too distorted to be classified.
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_fire_matrix)

dims = np.shape(TwoThSMt_fire)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)


# In[14]:


##get the path of the data set which are training data.
##These images should be labelled as no fire
import os
NoFire_listDir = os.listdir("../DataSet/Training/No_Fire")


# In[15]:


##glob library was used to get directory. It is used to show that data as has 
##been loaded
smjpegs_NoFire_Train = [f for f in glob.glob("../DataSet/Training/No_Fire/*.jpg")]
smjpegs_Fire_Train = [f for f in glob.glob("../DataSet/Training/Fire/*.jpg")]


# In[16]:


smjpegs_NoFire_Train


# In[17]:


smjpegs_Fire_Train


# In[18]:


set175 = [smj_NO_Train for smj_NO_Train in smjpegs_NoFire_Train if "lake_resized_lake" in smj_NO_Train]
print(set175)


# In[19]:


#https://www.kaggle.com/bkamphaus/exploratory-image-analysis
print(set175[0])


# In[20]:


TwoThSMt_No_fire = plt.imread('../DataSet/Training/Fire\\resized_frame23465.jpg')
dims = np.shape(TwoThSMt_fire)
print(dims)
plt.imshow(TwoThSMt_No_fire)


# In[21]:


img79_1, img79_2, img79_3, img79_4, img79_5 = [plt.imread(set175[n]) for n in range(1, 6)]


# In[22]:


img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

plt.figure(figsize=(8,10))
plt.imshow(img_list[4])
plt.show()


# In[24]:


##This class has been created to do some manipulation for further pre-processing
class MSImage():  
    def __init__(self, img):
        ##Image itself
        self.img = img
        ##the shape of the image
        self.dims = np.shape(img)
        ##Get the pixels and the number of colour of the image in a tuple
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    ##Returns the Image's matrix itself   
    @property
    def matrix(self):
        return self.mat
    ##Returns the Image itself      
    @property
    def image(self):
        return self.img
    
    ##Reshape the image data according the another data which will be shown in more detail.
    def to_matched_img(self, derived):
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))


# In[36]:


##created a matrix to partly deminstarte how 
_test = MSImage(TwoThSMt_No_fire)
##Return the shape of the image
print(_test.dims)
 ##Returns the Image's matrix itself  
print(_test.mat)

##After this line rest of the code is written to experiment with pre-processing. The more detail analysis done 
##Second python file name "Decision Making Data Walkthorugh Training"


# In[31]:


msi79_1 = MSImage(TwoThSMt_No_fire)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))


# In[47]:


##this function is taken from https://www.kaggle.com/bkamphaus/exploratory-image-analysis
##This function is used for basically normalization of the image or the matrix of the image
##it increases the values in the image matrix to increase the brightness of the image
def bnormalize(mat):
    ##much faster brightness normalization, since it's all vectorized quoted by
    ##writer of the https://www.kaggle.com/bkamphaus/exploratory-image-analysis
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm


# In[48]:


bnorm = bnormalize(msi79_1.matrix)
print(bnorm)


# In[49]:


##In this 
bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[38]:


print(bnorm)


# In[39]:


print(bnorm_img)


# In[40]:


msi79_1 = MSImage(img_list[0])
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))


# In[ ]:


smjpegs_Fire_Train[0]


# In[ ]:


img_fire= plt.imread(smjpegs_No_Fire_Train[9812])

plt.figure(figsize=(8,10))
plt.imshow(img_fire)
plt.show()


# In[ ]:


msi79_fire = MSImage(TwoThSMt_No_fire)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))


# In[ ]:



bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(TwoThSMt_No_fire)
plt.show()


# In[ ]:


from skimage.filters import sobel_h

# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(TwoThSMt_No_fire[:750,:750,2]))


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(msi79_1.matrix)
set144_0_pca = pca.transform(msi79_fire.matrix)
set144_0_pca_img = msi79_fire.to_matched_img(set144_0_pca)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')


# In[ ]:


from skimage import color

hsv = color.rgb2hsv(msi79_fire.image)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(msi79_fire.image, cmap="bone")
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,1], cmap='bone')
###plt.subplot(2,2,4)
##plt.imshow(hsv[:,:,2], cmap='bone')


# In[ ]:




