#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os

##In this python file, training dataset was explored


# In[2]:


##First picture taken from No fire altough it contains something resambles as a fire
no_fire_samp_1 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/lake_resized_lake_frame0.jpg")
dims_no_fire_1 = np.shape(no_fire_samp_1)
print(dims_no_fire_1)
pixel_matrix_no_fire_1 = np.reshape(no_fire_samp_1, (dims_no_fire_1[0] * dims_no_fire_1[1], dims_no_fire_1[2]))
print(np.shape(pixel_matrix_no_fire_1))
plt.imshow(no_fire_samp_1)


# In[53]:


##Second picture taken from No fire altough it contains fog which shouldn't be a problem
##Although there are images with fog and fire under Fire folder (images should be labelled as fire) 
##which indicated there might be misclassification between images like this which are not indicating forest fire 
##with forest fire. It can be seen in more detailed in pre-processing part
no_fire_samp_2 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/resized_frame6765.jpg")
dims_no_fire_2 = np.shape(no_fire_samp_2)
print(dims_no_fire_2)
pixel_matrix_no_fire_2 = np.reshape(no_fire_samp_2, (dims_no_fire_2[0] * dims_no_fire_2[1], dims_no_fire_2[2]))
print(np.shape(pixel_matrix_no_fire_2))
plt.imshow(no_fire_samp_2)


# In[4]:


##Third picture taken from No fire 
no_fire_samp_3 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/lake_resized_lake_frame6164.jpg")
dims_no_fire_3 = np.shape(no_fire_samp_3)
print(dims_no_fire_3)
pixel_matrix_no_fire_3 = np.reshape(no_fire_samp_3, (dims_no_fire_3[0] * dims_no_fire_3[1], dims_no_fire_3[2]))
print(np.shape(pixel_matrix_no_fire_3))
plt.imshow(no_fire_samp_3)


# In[5]:


##Fourth picture taken from No fire which contains sunset
no_fire_samp_4 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/lake_resized_lake_frame8325.jpg")
dims_no_fire_4 = np.shape(no_fire_samp_4)
print(dims_no_fire_4)
pixel_matrix_no_fire_4 = np.reshape(no_fire_samp_4, (dims_no_fire_4[0] * dims_no_fire_4[1], dims_no_fire_4[2]))
print(np.shape(pixel_matrix_no_fire_4))
plt.imshow(no_fire_samp_4)


# In[6]:


##Fifth picture taken with fog
no_fire_samp_5 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/No_Fire/resized_frame3590.jpg")
dims_no_fire_5 = np.shape(no_fire_samp_5)
print(dims_no_fire_5)
pixel_matrix_no_fire_5 = np.reshape(no_fire_samp_5, (dims_no_fire_5[0] * dims_no_fire_5[1], dims_no_fire_5[2]))
print(np.shape(pixel_matrix_no_fire_5))
plt.imshow(no_fire_samp_5)


# In[7]:


##Sixth picture taken from Fire folder 
fire_samp_0 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame0.jpg")
dims_fire_0 = np.shape(fire_samp_0)
print(dims_fire_0)
pixel_matrix_fire_0 = np.reshape(fire_samp_0, (dims_fire_0[0] * dims_fire_0[1], dims_fire_0[2]))
print(np.shape(pixel_matrix_fire_0))
plt.imshow(fire_samp_0)


# In[8]:


##Seventh picture taken from Fire folder with higher brightness
fire_samp_1 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame5.jpg")
dims_fire_1 = np.shape(fire_samp_1)
print(dims_fire_1)
pixel_matrix_fire_1 = np.reshape(fire_samp_1, (dims_fire_1[0] * dims_fire_1[1], dims_fire_1[2]))
print(np.shape(pixel_matrix_fire_1))
plt.imshow(fire_samp_1)


# In[9]:


##Eight picture taken from Fire folder contains some amount of fog and fires and multiple amount of trees 
##surrounding the fires
fire_samp_2 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame10954.jpg")
dims_fire_2 = np.shape(fire_samp_2)
print(dims_fire_2)
pixel_matrix_fire_2 = np.reshape(fire_samp_2, (dims_fire_2[0] * dims_fire_2[1], dims_fire_2[2]))
print(np.shape(pixel_matrix_fire_2))
plt.imshow(fire_samp_2)


# In[10]:


##Ninth picture taken from Fire folder contains some amount of fog and fires and multiple amount of trees 
##surrounding the fires. Fires are even smaller
fire_samp_3 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame14160.jpg")
dims_fire_3 = np.shape(fire_samp_3)
print(dims_fire_3)
pixel_matrix_fire_3 = np.reshape(fire_samp_3, (dims_fire_3[0] * dims_fire_3[1], dims_fire_3[2]))
print(np.shape(pixel_matrix_fire_3))
plt.imshow(fire_samp_3)


# In[11]:


##Tenth picture taken from Fire folder contains some amount of fog and fires and multiple amount of trees 
##surrounding the fires. Fires are even smallerfire_samp_4 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame14863.jpg")
dims_fire_4 = np.shape(fire_samp_4)
print(dims_fire_4)
pixel_matrix_fire_4 = np.reshape(fire_samp_4, (dims_fire_3[0] * dims_fire_3[1], dims_fire_3[2]))
print(np.shape(pixel_matrix_fire_4))
plt.imshow(fire_samp_4)


# In[12]:


fire_samp_5 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame6001.jpg")
dims_fire_5 = np.shape(fire_samp_5)
print(dims_fire_5)
pixel_matrix_fire_5 = np.reshape(fire_samp_5, (dims_fire_3[0] * dims_fire_3[1], dims_fire_3[2]))
print(np.shape(pixel_matrix_fire_5))
plt.imshow(fire_samp_5)


# In[13]:


fire_samp_6 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Training/Fire/resized_frame15664.jpg")
dims_fire_6 = np.shape(fire_samp_6)
print(dims_fire_6)
pixel_matrix_fire_6 = np.reshape(fire_samp_6, (dims_fire_3[0] * dims_fire_3[1], dims_fire_3[2]))
print(np.shape(pixel_matrix_fire_6))
plt.imshow(fire_samp_6)


# In[ ]:





# In[14]:


# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_no_fire_1)
dims = np.shape(no_fire_samp_1)
clustered_img_no_fire_1 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_no_fire_1)


# In[15]:


kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_no_fire_2)
dims = np.shape(no_fire_samp_2)
clustered_img_no_fire_2 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_no_fire_2)


# In[16]:


kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_no_fire_1)
dims = np.shape(no_fire_samp_3)
clustered_img_no_fire_3 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_no_fire_3)


# In[17]:


kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_fire_1)

dims = np.shape(fire_samp_1)
clustered_img_fire1 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_fire1)


# In[18]:


##Two pictures in line 18 and 19, look way to distorted to be classified. 
##Pictures both in the training and test folders will be more complicated than in picture line 7

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_fire_2)

dims = np.shape(fire_samp_2)
clustered_img_fire2 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_fire2)


# In[19]:


kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix_fire_3)

dims = np.shape(fire_samp_3)
clustered_img_fire3 = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img_fire3)


# In[20]:


#Get the directory
import os
NoFire_listDir = os.listdir("../DataSet/Training/No_Fire")


# In[21]:


smjpegs_NoFire_Train = [f for f in glob.glob("../DataSet/Training/No_Fire/*.jpg")]
smjpegs_Fire_Train = [f for f in glob.glob("../DataSet/Training/Fire/*.jpg")]


# In[22]:


smjpegs_NoFire_Train


# In[23]:


smjpegs_Fire_Train


# In[24]:


No_fires_pic = plt.imread('../DataSet/Training/No_Fire\\resized_frame6773.jpg')
dims = np.shape(No_fires_pic)
print(dims)
plt.imshow(No_fires_pic)


# In[25]:


class MSImage():
   
    
    def __init__(self, img):

        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_matched_img(self, derived):
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))


# In[26]:


msi79_1 = MSImage(fire_samp_3)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))


# In[27]:


def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm


# In[28]:


msi79_1


# In[29]:


##MSI Class was explained in Decision Making Data Walkthorugh file
##Picture in line 2 brightness have been increase. The fire in the picture can be seen
##in the picture as red. although it doesn't mean classifier will classify this image as "Fire" 
msi79_no_fire_1 = MSImage(no_fire_samp_1)
print(np.shape(msi79_no_fire_1.matrix))
print(np.shape(msi79_no_fire_1.img))

bnorm = bnormalize(msi79_no_fire_1.matrix)
bnorm_img = msi79_no_fire_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[30]:


##Picture in line 3 brightness have been increased. 
msi79_no_fire_2 = MSImage(no_fire_samp_2)
print(np.shape(msi79_no_fire_2.matrix))
print(np.shape(msi79_no_fire_2.img))

bnorm = bnormalize(msi79_no_fire_2.matrix)
bnorm_img = msi79_no_fire_2.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[31]:


##Picture in line 4 rightness have been increased. 
msi79_no_fire_3 = MSImage(no_fire_samp_3)
print(np.shape(msi79_no_fire_3.matrix))
print(np.shape(msi79_no_fire_3.img))

bnorm = bnormalize(msi79_no_fire_3.matrix)
bnorm_img = msi79_no_fire_3.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[32]:


##Picture in line 5 rightness have been increased. The sun in the picture in line 5
##have yellow colour which is different then fire when the brightness increased.

msi79_no_fire_4 = MSImage(no_fire_samp_4)
print(np.shape(msi79_no_fire_4.matrix))
print(np.shape(msi79_no_fire_4.img))

bnorm = bnormalize(msi79_no_fire_4.matrix)
bnorm_img = msi79_no_fire_4.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[33]:


##Picture in line 6 rightness have been increased. Image seems to be 
##distorted due to fog


msi79_no_fire_5 = MSImage(no_fire_samp_5)
print(np.shape(msi79_no_fire_5.matrix))
print(np.shape(msi79_no_fire_5.img))

bnorm = bnormalize(msi79_no_fire_5.matrix)
bnorm_img = msi79_no_fire_5.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[34]:


##Picture in line 7, fire seems to have distinct colours from other
##object with higher brightness.

msi79_fire_1 = MSImage(fire_samp_0)
print(np.shape(msi79_fire_1.matrix))
print(np.shape(msi79_fire_1.img))

bnorm = bnormalize(msi79_fire_1.matrix)
bnorm_img = msi79_fire_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[35]:


##Picture in line 8, fire seems to have distinct colours from other
##object with higher brightness. However, fire seems to be less visible because the original image in the line 8 
##have more brightness

msi79_fire_2 = MSImage(fire_samp_1)
print(np.shape(msi79_fire_2.matrix))
print(np.shape(msi79_fire_2.img))

bnorm = bnormalize(msi79_fire_2.matrix)
bnorm_img = msi79_fire_2.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[36]:


##MSI Class was explained in Decision Making Data Walkthorugh file
##Picture in line 9 there are multiple fires. Fires can be seen with their deep orange colours


msi79_fire_3 = MSImage(fire_samp_2)
print(np.shape(msi79_fire_3.matrix))
print(np.shape(msi79_fire_3.img))

bnorm = bnormalize(msi79_fire_3.matrix)
bnorm_img = msi79_fire_3.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[37]:


##MSI Class was explained in Decision Making Data Walkthorugh file
##Picture in line  there are multiple fires. Fires can be seen with their deep orange colours
##Below four images from line 37-40, red points are originally fires. Because of fire have so
##much red and deep orange colour, it is different from sunset with incrased 

msi79_fire_4 = MSImage(fire_samp_3)
print(np.shape(msi79_fire_4.matrix))
print(np.shape(msi79_fire_4.img))

bnorm = bnormalize(msi79_fire_4.matrix)
bnorm_img = msi79_fire_4.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[38]:


msi79_fire_5 = MSImage(fire_samp_4)
print(np.shape(msi79_fire_5.matrix))
print(np.shape(msi79_fire_5.img))

bnorm = bnormalize(msi79_fire_5.matrix)
bnorm_img = msi79_fire_5.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[39]:


msi79_fire_6 = MSImage(fire_samp_5)
print(np.shape(msi79_fire_6.matrix))
print(np.shape(msi79_fire_6.img))

bnorm = bnormalize(msi79_fire_6.matrix)
bnorm_img = msi79_fire_6.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[40]:


msi79_fire_7 = MSImage(fire_samp_6)
print(np.shape(msi79_fire_7.matrix))
print(np.shape(msi79_fire_7.img))

bnorm = bnormalize(msi79_fire_7.matrix)
bnorm_img = msi79_fire_7.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[41]:


##The preprocessing below is done for experimentation as an alternative to increasing the brightness.
##The brighther dots should indicate the fire in the pictures.
from skimage import color

hsv = color.rgb2hsv(msi79_no_fire_1.image)
plt.imshow(msi79_no_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[42]:


hsv = color.rgb2hsv(msi79_no_fire_2.image)
plt.imshow(msi79_no_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[43]:


hsv = color.rgb2hsv(msi79_no_fire_3.image)
plt.imshow(msi79_no_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[44]:


hsv = color.rgb2hsv(msi79_no_fire_4.image)
plt.imshow(msi79_no_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[45]:


hsv = color.rgb2hsv(msi79_no_fire_5.image)
plt.imshow(msi79_no_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[46]:


hsv = color.rgb2hsv(msi79_fire_1.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[47]:


hsv = color.rgb2hsv(msi79_fire_2.image)
plt.imshow(msi79_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[48]:


hsv = color.rgb2hsv(msi79_fire_3.image)
plt.imshow(msi79_fire_3.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[49]:


hsv = color.rgb2hsv(msi79_fire_4.image)
plt.imshow(msi79_fire_4.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[50]:


hsv = color.rgb2hsv(msi79_fire_5.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[55]:


hsv = color.rgb2hsv(msi79_fire_6.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[56]:


hsv = color.rgb2hsv(msi79_fire_7.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[ ]:





# In[ ]:




