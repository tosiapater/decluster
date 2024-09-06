# -*- coding: utf-8 -*-
"""
run cut_out first
gets class info from color dots
"""
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2



save_dir = "outputs_with_classes"
imgfolderpath = "cut_out_output"
reffolderparh = "points"
smalllist = []
pngimages = [f for f in os.listdir(imgfolderpath) if f.endswith(".png")]
for filename in pngimages:
    
    # get location of the patch
    im = Image.open(os.path.join(imgfolderpath, filename))
    width, height = im.size
    
    # if np.min(width)<40: 
    #     smalllist.append(filename)
    #     continue
    
    basename = filename.split("_")[0]
    miny = int(filename.split("_")[-1].split(".")[0])
    minx = int(filename.split("_")[1][8:])
    
    
    # get class info
    folder = basename[:4]
    margin = 11
    refim = cv2.imread(os.path.join(reffolderparh,folder, basename+".png"))
    refpatch =refim[miny+margin:miny+height-margin,minx+margin:minx+width-margin]

    # plt.imshow(refpatch)
    # plt.axis('off')
    # plt.show()

    red = (0,0,255) 
    green = (0,255,0)
    
    colors = {
          "HSIL": (0,0,255),
          "LSIL": (255,0,255),
          "NILM": (0,255,0)
        }
    

    savedflag = 0
    for color in colors:
        ref_col = cv2.inRange(refpatch, colors[color], colors[color])
        if np.sum(ref_col)>10:
            im.save(os.path.join(save_dir, color, filename))
            savedflag=1
            continue
    if savedflag==0:
        im.save(os.path.join(save_dir, "UNKN", filename))
        
            
            

    
    
    
    
