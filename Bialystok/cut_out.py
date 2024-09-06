# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import keras
from PIL import Image

""" 
before running generate watershed maps

loads segmentation counturs and cuts images accordingly. 

note: map loads in cv, image in PIL 
""" 


save_dir = "cut_out_output"
mapfolderpath = "waterseeded"
imgfolderpath = "clusters"
errorlist = []

hsilimages = ["HSIL\\"+f for f in os.listdir(imgfolderpath+"\\HSIL") if f.endswith(".png")]
lsilimages = ["LSIL\\"+f for f in os.listdir(imgfolderpath+"\\LSIL") if f.endswith(".png")]
nlimimages = ["NILM\\"+f for f in os.listdir(imgfolderpath+"\\NILM") if f.endswith(".png")]

clusterimages = hsilimages + lsilimages + nlimimages

for clusterpath in clusterimages:
    
    basename = clusterpath.split(".")[0].split("\\")[-1]
    ext = clusterpath.split(".")[-1]
    foldername = clusterpath[:4]
    oldfoldername = basename[:4]
    
    mappath = clusterpath.split("\\")[-1] + "_maskWaterSeeded.png"
    map1 = cv2.imread(os.path.join(mapfolderpath, mappath))
    if map1 is None:
        errorlist.append(basename)
        continue
    imagebw = map1[:,:,0]

    
    image = np.array(Image.open(os.path.join(imgfolderpath, foldername,basename+'.'+ext)))
    
    contours, hierarchy = cv2.findContours(imagebw,  
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        errorlist.append(basename)
        continue

      
    orgshape = map1.shape
    cntrnum = 0
    margin = 7 

    if len(contours) > 1:
        for inx, contour in enumerate(contours):
            
            
            corners = [max(contour[:,0,0]), max(contour[:,0,1]), min(contour[:,0,0]), min(contour[:,0,1])]        
            markersize = [corners[0]-corners[2],corners[1]-corners[3]]

            if min(markersize)<35: # add margin
                continue
            
            # add margin
            if corners[2]<=margin:
                corners[2]==0
            else:
                corners[2] = corners[2]-margin     
            if corners[3]<=margin:
                corners[3]==0
            else:
                corners[3] = corners[3]-margin
            if corners[0]+margin >= orgshape[1]:
                corners[0] = orgshape[1]
            else:
                corners[0] = corners[0] + margin
            if corners[1]+margin >= orgshape[0]:
                corners[1] = orgshape[0]
            else:
                corners[1] = corners[1] + margin    
    
            markercell = image[corners[3]:corners[1],corners[2]:corners[0]]
            markercell = keras.utils.array_to_img(markercell)

            markercell.save(os.path.join(save_dir, basename+"_mincornr"+str(corners[2])+"_"+str(corners[3])+"."+ext))
            cntrnum = cntrnum+1
    else:
        if hierarchy[0,0,3] ==-1: continue
        contour = contours[0]
        corners = [max(contour[:,0,0]), max(contour[:,0,1]), min(contour[:,0,0]), min(contour[:,0,1])]        
        markersize = [corners[0]-corners[2],corners[1]-corners[3]]

        if corners[2]<=margin:
            corners[2]==0
        else:
            corners[2] = corners[2]-margin     
        if corners[3]<=margin:
            corners[3]==0
        else:
            corners[3] = corners[3]-margin
        if corners[0]+margin >= orgshape[1]:
            corners[0] = orgshape[1]
        else:
            corners[0] = corners[0] + margin
        if corners[1]+margin >= orgshape[0]:
            corners[1] = orgshape[0]
        else:
            corners[1] = corners[1] + margin 
        markercell = image[corners[3]:corners[1],corners[2]:corners[0]]

            
        markercell = keras.utils.array_to_img(markercell)
        markercell.save(os.path.join(save_dir, basename+"_mincornr"+str(corners[2])+"_"+str(corners[3])+"."+ext))
        
print(errorlist)
