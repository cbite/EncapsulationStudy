# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:55:46 2019

@author: 20190294

"""

import numpy as np
import pandas as pd
import PIL, codecs, json, array, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import cv2
from PIL import Image 
##Load surfaces Definiti

M_L_data=pd.read_csv("Featureidx_with_labels.csv", sep=";")

#M_L_data.shape[]

#M_L_data=M_L_data[M_L_data.Label==0]

IMAGE_SIZE=28
NUM_CHANNELS=1

MacrophageSurfData=np.zeros((M_L_data.shape[0]*12,IMAGE_SIZE,IMAGE_SIZE)).astype(np.float32)

MacrophageSurfDataL=np.zeros((M_L_data.shape[0]*12,1)).astype(np.float32)


##Load Surfaces stack yo the data set and multiply it and assign labels

rotation_angles = [0,90,180,270]

def load_macrophage_surf_data(FeatureLabelData):
    
    sk=0
    #FeatureLabelData= M_L_data[M_L_data.Label==0]
    ##Load all images
    for i in FeatureLabelData.FeatureIdx.values:
        print (sk)
        print (i)
        FeatImg = PIL.Image.open('Surface_Images/Pattern_FeatureIdx_{}.bmp'.format(i)).convert("L")
        FeatImg=FeatImg.resize([28,28])
        FeatLabel=M_L_data.Label.loc[M_L_data.FeatureIdx==i]
        ##rotate images
        
        for angle in rotation_angles:
            im0=FeatImg
            im1=im0.rotate(angle)
            im2=im1.transpose(Image.FLIP_LEFT_RIGHT)
            im3=im1.transpose(Image.FLIP_TOP_BOTTOM)
            
            for images in [im1,im2,im3]:
#                images.setflags(write=1)
#                images[images>0]=1
                MacrophageSurfData[sk,:]=np.asarray(images)
                MacrophageSurfDataL[sk,:]=np.int(FeatLabel)
                
                sk+=1
    return (MacrophageSurfData,MacrophageSurfDataL)
        
X,Y=load_macrophage_surf_data(M_L_data)

#file_path = "MacrophageSurfaceData.json"
#json.dump(X, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


