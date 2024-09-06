# -*- coding: utf-8 -*-
"""
Draw results of seperated cell segmentation on cluster image
run cut_out first
"""

import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from vgg import create_model # local
from keras.applications.vgg16 import preprocess_input
import cv2
import matplotlib.pyplot as plt


loc_path =  'waterseeded'
cluster_path = 'clusters'
save_dir = 'results\\'
blob_path = 'masks'


hsilimages = ["HSIL\\"+f for f in os.listdir(cluster_path+"\HSIL") if f.endswith(".png")]
lsilimages = ["LSIL\\"+f for f in os.listdir(cluster_path+"\LSIL") if f.endswith(".png")]

clusterimages = hsilimages + lsilimages


colors = {
      0: (255, 0, 0), # HSIL
      1: (255,0,255), # LSIL
      2: (0,255,0), # NILM
      # 3: (255,255,255) # artifact
    }

# VGG16 settings

vgg_model = create_model((224, 224, 3), 3)
vgg_model.load_weights(r"../weights/VGG16_3_class.hdf5") # with gan

errorlist = []
for filename in clusterimages[:1]:
    
    im = Image.open(os.path.join(cluster_path, filename))
    name = filename.split("\\")[1].split(".")[0]
    patch_names = glob.glob(loc_path + "\\"+name+"*.png")
    
    if patch_names == []: 
        errorlist.append(name)
        continue
    
    patches_resized = []
    patches_loc = []
    for patch_name in patch_names:
        patch = Image.open(patch_name)
        width, height = patch.size
        miny = int(patch_name.split("_")[-1].split(".")[0])
        minx = int(patch_name.split("\\")[-1].split("_")[1][8:])
        
        patches_loc.append([minx,miny,minx+height,miny+width])
        patches_resized.append(np.array(patch.resize((224,224))))
        
    X = preprocess_input(np.array(patches_resized))
    preds = vgg_model.predict(X)
    pred_classes = np.argmax(preds, axis=1)
    
    for inx, pred_class in enumerate(pred_classes):
         
        img1 = ImageDraw.Draw(im)   
        img1.rectangle(patches_loc[inx], outline =colors[pred_class], width=4) 
        
    oldfolder = name[:4]
    blob = cv2.imread(os.path.join(blob_path,oldfolder, name+".png"), cv2.IMREAD_GRAYSCALE) 
    blob_bin = (blob>0).astype("uint8")
    blob_contours,_ = cv2.findContours(blob_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    im_cont = np.array(im)
    cv2.drawContours(im_cont, blob_contours, -1, (0, 0, 0), 2)
    im = Image.fromarray(im_cont)
    
    plt.figure()
    plt.imshow(im_cont)
    plt.axis('off')
    plt.show()

    # im.save(save_dir+name+".png")
    