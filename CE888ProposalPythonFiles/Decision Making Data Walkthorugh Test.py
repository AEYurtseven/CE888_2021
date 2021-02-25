#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os


# In[2]:


no_fire_samp_1 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/No_Fire/resized_test_nofire_frame0.jpg")
dims_no_fire_1 = np.shape(no_fire_samp_1)
print(dims_no_fire_1)
pixel_matrix_no_fire_1 = np.reshape(no_fire_samp_1, (dims_no_fire_1[0] * dims_no_fire_1[1], dims_no_fire_1[2]))
print(np.shape(pixel_matrix_no_fire_1))
plt.imshow(no_fire_samp_1)


# In[3]:


no_fire_samp_2 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/No_Fire/resized_test_nofire_frame205.jpg")
dims_no_fire_2 = np.shape(no_fire_samp_2)
print(dims_no_fire_2)
pixel_matrix_no_fire_2 = np.reshape(no_fire_samp_2, (dims_no_fire_2[0] * dims_no_fire_2[1], dims_no_fire_2[2]))
print(np.shape(pixel_matrix_no_fire_2))
plt.imshow(no_fire_samp_2)


# In[4]:


no_fire_samp_3 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/No_Fire/resized_test_nofire_frame260.jpg")
dims_no_fire_3 = np.shape(no_fire_samp_3)
print(dims_no_fire_3)
pixel_matrix_no_fire_3 = np.reshape(no_fire_samp_3, (dims_no_fire_3[0] * dims_no_fire_3[1], dims_no_fire_3[2]))
print(np.shape(pixel_matrix_no_fire_3))
plt.imshow(no_fire_samp_3)


# In[5]:


no_fire_samp_4 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/No_Fire/resized_test_nofire_frame485.jpg")
dims_no_fire_4 = np.shape(no_fire_samp_4)
print(dims_no_fire_4)
pixel_matrix_no_fire_4 = np.reshape(no_fire_samp_4, (dims_no_fire_4[0] * dims_no_fire_4[1], dims_no_fire_4[2]))
print(np.shape(pixel_matrix_no_fire_4))
plt.imshow(no_fire_samp_4)


# In[6]:


no_fire_samp_5 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/No_Fire/resized_test_nofire_frame2621.jpg")
dims_no_fire_5 = np.shape(no_fire_samp_5)
print(dims_no_fire_5)
pixel_matrix_no_fire_5 = np.reshape(no_fire_samp_5, (dims_no_fire_5[0] * dims_no_fire_5[1], dims_no_fire_5[2]))
print(np.shape(pixel_matrix_no_fire_5))
plt.imshow(no_fire_samp_5)


# In[7]:


fire_samp_0 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame0.jpg")
dims_fire_0 = np.shape(fire_samp_0)
print(dims_fire_0)
pixel_matrix_fire_0 = np.reshape(fire_samp_0, (dims_fire_0[0] * dims_fire_0[1], dims_fire_0[2]))
print(np.shape(pixel_matrix_fire_0))
plt.imshow(fire_samp_0)


# In[10]:


fire_samp_1 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame222.jpg")
dims_fire_1 = np.shape(fire_samp_1)
print(dims_fire_1)
pixel_matrix_fire_1 = np.reshape(fire_samp_1, (dims_fire_1[0] * dims_fire_1[1], dims_fire_1[2]))
print(np.shape(pixel_matrix_fire_1))
plt.imshow(fire_samp_1)


# In[11]:


fire_samp_2 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame536.jpg")
dims_fire_2 = np.shape(fire_samp_2)
print(dims_fire_2)
pixel_matrix_fire_2 = np.reshape(fire_samp_2, (dims_fire_2[0] * dims_fire_2[1], dims_fire_2[2]))
print(np.shape(pixel_matrix_fire_2))
plt.imshow(fire_samp_2)


# In[12]:


fire_samp_3 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame661.jpg")
dims_fire_3 = np.shape(fire_samp_3)
print(dims_fire_3)
pixel_matrix_fire_3 = np.reshape(fire_samp_3, (dims_fire_3[0] * dims_fire_3[1], dims_fire_3[2]))
print(np.shape(pixel_matrix_fire_3))
plt.imshow(fire_samp_3)


# In[18]:



fire_samp_4 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame3398.jpg")
dims_fire_4 = np.shape(fire_samp_4)
print(dims_fire_4)
pixel_matrix_fire_4 = np.reshape(fire_samp_4, (dims_fire_4[0] * dims_fire_4[1], dims_fire_4[2]))
print(np.shape(pixel_matrix_fire_4))
plt.imshow(fire_samp_4)


# In[19]:


fire_samp_5 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame1619.jpg")
dims_fire_5 = np.shape(fire_samp_5)
print(dims_fire_5)
pixel_matrix_fire_5 = np.reshape(fire_samp_5, (dims_fire_5[0] * dims_fire_5[1], dims_fire_5[2]))
print(np.shape(pixel_matrix_fire_5))
plt.imshow(fire_samp_5)


# In[20]:


fire_samp_6 = plt.imread("C:/Users/erdal/Desktop/Essex Ders/Decision Making/DataSet/Test/Fire/resized_test_fire_frame3398.jpg")
dims_fire_6 = np.shape(fire_samp_6)
print(dims_fire_6)
pixel_matrix_fire_6 = np.reshape(fire_samp_6, (dims_fire_6[0] * dims_fire_6[1], dims_fire_6[2]))
print(np.shape(pixel_matrix_fire_6))
plt.imshow(fire_samp_6)


# In[21]:


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
    
    def to_flat_img(self, derived):
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))


# In[22]:


def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm


# In[23]:


msi79_no_fire_1 = MSImage(no_fire_samp_1)
print(np.shape(msi79_no_fire_1.matrix))
print(np.shape(msi79_no_fire_1.img))

bnorm = bnormalize(msi79_no_fire_1.matrix)
bnorm_img = msi79_no_fire_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[24]:


msi79_no_fire_2 = MSImage(no_fire_samp_2)
print(np.shape(msi79_no_fire_2.matrix))
print(np.shape(msi79_no_fire_2.img))

bnorm = bnormalize(msi79_no_fire_2.matrix)
bnorm_img = msi79_no_fire_2.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[25]:


msi79_no_fire_3 = MSImage(no_fire_samp_3)
print(np.shape(msi79_no_fire_3.matrix))
print(np.shape(msi79_no_fire_3.img))

bnorm = bnormalize(msi79_no_fire_3.matrix)
bnorm_img = msi79_no_fire_3.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[26]:


msi79_no_fire_4 = MSImage(no_fire_samp_4)
print(np.shape(msi79_no_fire_4.matrix))
print(np.shape(msi79_no_fire_4.img))

bnorm = bnormalize(msi79_no_fire_4.matrix)
bnorm_img = msi79_no_fire_4.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[27]:


msi79_no_fire_5 = MSImage(no_fire_samp_5)
print(np.shape(msi79_no_fire_5.matrix))
print(np.shape(msi79_no_fire_5.img))

bnorm = bnormalize(msi79_no_fire_5.matrix)
bnorm_img = msi79_no_fire_5.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[28]:


from skimage import color

hsv = color.rgb2hsv(msi79_no_fire_1.image)
plt.imshow(msi79_no_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[42]:


hsv = color.rgb2hsv(msi79_no_fire_2.image)
plt.imshow(msi79_no_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[39]:


hsv = color.rgb2hsv(msi79_no_fire_3.image)
plt.imshow(msi79_no_fire_3.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[36]:


hsv = color.rgb2hsv(msi79_no_fire_4.image)
plt.imshow(msi79_no_fire_4.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[33]:


hsv = color.rgb2hsv(msi79_no_fire_5.image)
plt.imshow(msi79_no_fire_5.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[43]:


msi79_fire_1 = MSImage(fire_samp_0)
print(np.shape(msi79_fire_1.matrix))
print(np.shape(msi79_fire_1.img))

bnorm = bnormalize(msi79_fire_1.matrix)
bnorm_img = msi79_fire_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[44]:


msi79_fire_2 = MSImage(fire_samp_1)
print(np.shape(msi79_fire_2.matrix))
print(np.shape(msi79_fire_2.img))

bnorm = bnormalize(msi79_fire_2.matrix)
bnorm_img = msi79_fire_2.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[45]:


msi79_fire_3 = MSImage(fire_samp_2)
print(np.shape(msi79_fire_3.matrix))
print(np.shape(msi79_fire_3.img))

bnorm = bnormalize(msi79_fire_3.matrix)
bnorm_img = msi79_fire_3.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[46]:


msi79_fire_4 = MSImage(fire_samp_3)
print(np.shape(msi79_fire_4.matrix))
print(np.shape(msi79_fire_4.img))

bnorm = bnormalize(msi79_fire_4.matrix)
bnorm_img = msi79_fire_4.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[47]:


msi79_fire_5 = MSImage(fire_samp_4)
print(np.shape(msi79_fire_5.matrix))
print(np.shape(msi79_fire_5.img))

bnorm = bnormalize(msi79_fire_5.matrix)
bnorm_img = msi79_fire_5.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[48]:


msi79_fire_6 = MSImage(fire_samp_5)
print(np.shape(msi79_fire_6.matrix))
print(np.shape(msi79_fire_6.img))

bnorm = bnormalize(msi79_fire_6.matrix)
bnorm_img = msi79_fire_6.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[49]:


msi79_fire_7 = MSImage(fire_samp_6)
print(np.shape(msi79_fire_7.matrix))
print(np.shape(msi79_fire_7.img))

bnorm = bnormalize(msi79_fire_7.matrix)
bnorm_img = msi79_fire_7.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[53]:


hsv = color.rgb2hsv(msi79_fire_1.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[54]:


hsv = color.rgb2hsv(msi79_fire_2.image)
plt.imshow(msi79_fire_2.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[55]:


hsv = color.rgb2hsv(msi79_fire_3.image)
plt.imshow(msi79_fire_3.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[56]:


hsv = color.rgb2hsv(msi79_fire_4.image)
plt.imshow(msi79_fire_4.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[57]:


hsv = color.rgb2hsv(msi79_fire_5.image)
plt.imshow(msi79_fire_5.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[58]:


hsv = color.rgb2hsv(msi79_fire_6.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[59]:


hsv = color.rgb2hsv(msi79_fire_7.image)
plt.imshow(msi79_fire_1.image, cmap="bone")
plt.imshow(hsv[:,:,1], cmap='bone')


# In[ ]:




