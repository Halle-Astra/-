#!/usr/bin/env python
# coding: utf-8

# In[3]:


from matplotlib import pyplot as plt 
import numpy as np 
from PIL import Image 
import cv2 
import os 


def cvt2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[6]:


def mean2zero(img):
    return img-img.mean()


# In[7]:


def img2bi(img,thre = -10,rev = False):
    img_bk = img.copy()
    img[img_bk>thre]=1
    img[img_bk<thre]=0
    return img


# # 修正左上角的暗光影响

# In[10]:


def img2bi(img,thre = -10,rev = True):
    img_bk = img.copy()
    img[img_bk>thre]=1
    img[img_bk<thre]=0
    if rev:
        img[:12,:30] = 1 
    return img

# In[15]:


def process(img1):
    img1 = cvt2gray(img1)
    img1 = mean2zero(img1)
    img1 = img2bi(img1)
    return img1


# In[ ]:




