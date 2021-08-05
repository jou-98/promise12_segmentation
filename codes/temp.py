#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
from collections import defaultdict
import os, pickle, sys
import shutil
from functools import partial
# from itertools import izip

import cv2
import numpy as np
# from scipy.misc import imresize
from cv2 import resize as imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist

from models import *
from metrics import dice_coef, dice_coef_loss
from augmenters import *


img_shapes =    {'23': (30, 512, 512), '19': (30, 512, 512), '22': (30, 512, 512), 
                '25': (30, 512, 512), '24': (30, 512, 512), '26': (30, 512, 512), 
                '21': (31, 320, 320), '20': (30, 512, 512)}

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def data_to_array(img_rows, img_cols):


    fileList =  os.listdir('../data/train/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList = sorted(fileList) # fileList.sort() 
    # Train : Val = 7 : 43
    val_list = [7,14,21,28,35,42,49]
    train_list = list( set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

        for filename in filtered:

            itkimage = sitk.ReadImage('../data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )

            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs )

        images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = masks.astype(int)

        #Smooth images using CurvatureFlow
        images = smooth_images(images)

        if count==0:
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save('../data/X_train.npy', images)
            np.save('../data/y_train.npy', masks)
        elif count==1:
            images = (images - mu)/sigma

            np.save('../data/X_val.npy', images)
            np.save('../data/y_val.npy', masks)
        count+=1

    fileList =  os.listdir('../Dataset/')
    fileList = filter(lambda x: '.bin' in x, fileList)
    fileList = sorted(fileList) # fileList.sort() 
    n_imgs=[]
    images=[]
    masks=[]
    for filename in fileList:
        idx = filename[:-8] if 'seg' in filename.lower() else filename[:-4]
        imgs = np.fromfile('../Dataset/'+filename,dtype=np.float32).reshape(img_shapes[idx])
        #print(imgs.shape)
        #imgs = sitk.GetArrayFromImage(itkimage)
        if 'seg' in filename.lower():
            imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
            masks.append(imgs)
        else:
            imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
            images.append(imgs)
            n_imgs.append( len(imgs) )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
    masks = masks.astype(int)
    mu = np.mean(images)
    sigma = np.std(images)
    images = smooth_images(images)
    images = (images - mu)/sigma
    np.save('../data/test_x.npy', images)
    np.save('../data/y_test.npy', masks)
    np.save('../data/test_n_imgs.npy', np.array(n_imgs) )


def load_data():

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')


    return X_train, y_train, X_val, y_val

data_to_array(img_rows=256, img_cols=256)