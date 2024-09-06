# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import cv2

from watershed_cutout import watershed # local
from mask_to_countours import mask2contours # local

# getclasspointcric3class(name_list[inx], np.zeros_like(roimarked), folderann, dotsize=0)
def getclasspointcric3class(name, roi, dictlist, dotsize=20, drawcontour=None):
    """
    works only for HSIL/LSIL/NILM classiciation. draws points on the roi and a blank image

    """
    name = name +'.png'
    colorlegend = {  # BRG colors
        "ASC-H" : (0,0,255),
        "HSIL": (0,0,255),
        "ASC-US": (255,0,255),
        "LSIL": (255,0,255),
        "Negative for intraepithelial lesion": (0,255,0),
        "SCC": (0,0,255)
        }
    only_points = np.zeros_like(roi)
    
    celldir = [element for element in dictlist if element['image_name'] == name][0]
    
    for cell in celldir['classifications']:
        location = cell['nucleus_x'], cell['nucleus_y']
        color = colorlegend[cell['bethesda_system']]

        cv2.circle(roi, location, dotsize, color, -1)
        cv2.circle(only_points, location, dotsize, color, -1)
    
    
    return [roi, only_points]


def getclasspointcricbin(name, roi, dictlist, dotsize=20, drawcontour=None):
    """
    works only for SIL vs NILM classiciation. draws points on the roi and a blank image

    """
    
    colorlegend = {  # BRG colors
        "ASC-H" : (0,0,255),
        "HSIL": (0,0,255),
        "ASC-US": (0,0,255),
        "LSIL": (0,0,255),
        "Negative for intraepithelial lesion": (0,255,0),
        "SCC": (0,0,255)
        }
    only_points = np.zeros_like(roi)
    
    name = name +'.png'    
    celldir = [element for element in dictlist if element['image_name'] == name][0]
    
    for cell in celldir['classifications']:
        location = cell['nucleus_x'], cell['nucleus_y']
        color = colorlegend[cell['bethesda_system']]

        cv2.circle(roi, location, dotsize, color, -1)
        cv2.circle(only_points, location, dotsize, color, -1)
    
    
    return [roi, only_points]


def getclasspointcric(name, roi, dictlist, dotsize=20, drawcontour=None):
    """
    6 class classification. draws points on the roi and a blank image

    """
    
    colorlegend = {  # BRG colors
        "ASC-H" : (0,165,255),
        "ASC-US": (180,105,255),
        "HSIL": (0,0,255),
        "LSIL": (255,0,255),
        "Negative for intraepithelial lesion": (0,255,0),
        "SCC": (255,255,255)
        }
    only_points = np.zeros_like(roi)
    
    celldir = [element for element in dictlist if element['image_name'] == name][0]
    
    for cell in celldir['classifications']:
        location = cell['nucleus_x'], cell['nucleus_y']
        color = colorlegend[cell['bethesda_system']]

        cv2.circle(roi, location, dotsize, color, -1)
        cv2.circle(only_points, location, dotsize, color, -1)
    
    
    return [roi, only_points]

def getclasspointjson(jsondict, roi, dotsize=20): # TODO: unfunished
    """
    works only for HSIL/LSIL/NILM classiciation.
    color dict for all classes in nucleilocjson

    """
    
    colorlegend = {  # BRG colors
        "endocervix" :  (0,255,0),# (207, 157, 22),
        "superficial" : (0,255,0),#(255, 252,3),
        "intermediate": (0,255,0),#(19, 255, 12),
        "parabasal":    (0,255,0),#(17, 249,235),
        #"neutrophil":   (236,0,0),
        "metaplastic":  (0,255,0),#(136, 141,135),
        "hsil":         (0,0,255), #(164, 7, 2),
        "asch":         (0,0,255), #(232, 103, 62),
        "lsil":         (255,0,255), #(238, 14, 218),
        "koilocytes":   (255,0,255), #(252, 141, 223),
        "ascus":        (255,0,255), #(138, 77, 117),
        "parakeratosis":(0,0,255), # (255, 255, 255) ?????????????
        }
    
    only_points = np.zeros_like(roi)
    
    for colorkey in colorlegend:
        color = colorlegend[colorkey]
        locationsdot = jsondict[colorkey]
        
        for location in locationsdot:
            
            cv2.circle(roi, tuple(location), dotsize, color, -1)
            cv2.circle(only_points, tuple(location), dotsize, color, -1)
    
    return [roi, only_points]
    
    

def getbinclasspoint(namenum, roi, foldermap, dotsize=20, drawcontour=None):
    #if contour == None draws a dot, else a countour
    
    only_points = np.zeros_like(roi) # for an image with only points
    name = namenum[:-1]
    num = namenum[-1:]
    
    # color_dict  = {
    #          # 'parabasal':   (235,249,17),
    #          'HSIL':        (2,7,164),
    #          'ASCH':        (5, 167, 247), #105, 3, 143 5, 167, 247
    #          'LSIL':        (234,104,223),
    #          'ASCUS':       (237, 157, 38),
    #          # 'metaplastic': (135,141,136),
    #          'superficial': (12,255,19)} # incl intermediate

    color_dict  = {
        "ASC-H" : (0,165,255),
        "ASC-US": (180,105,255),
        "HSIL": (0,0,255),
        "LSIL": (255,0,255),
        "NILM": (0,255,0),
        # "Artifacts": (0,255,255)
        }
    ## wczytanie map i wyciągnięcie centrów
    mapimgs = [i for i in glob.glob(os.path.join(foldermap,name+'*')) if i[-5]==num]
    # print(name)
    
    for mapim in mapimgs:
        mapname = os.path.basename(mapim)
        classtype = mapname[21:-5]
        # print(classtype)
           # find color
           # print(classtype.lower())
        if classtype.lower() == "powierzchowne" or classtype.lower() == "powierzchniowe" or classtype.lower() == "posrednie" or classtype.lower() == "pośrednie":
            color = color_dict['NILM']
        elif classtype.lower() == "przypodstawne":
            color = color_dict['NILM']
        elif classtype.lower() == "metaplastyczne" or classtype.lower() == "metaplast":
            color = color_dict['NILM']
        elif classtype.lower() == "HSIL".lower():
            color = color_dict['HSIL']
        elif classtype.lower() == "ASC-H".lower():
            color = color_dict['HSIL']
        elif classtype.lower() == "LSIL".lower() or classtype.lower() == "LSIL nie-ko".lower() or classtype.lower() == "koilocyty":
            color = color_dict['HSIL']
        elif classtype.lower() == "ASC-US".lower() or classtype.lower() == "ASC-US".lower():
            color = color_dict['HSIL']
        else: 
            color = None
                # break
           # print(color)
        
        if classtype == "pojedyncze" or classtype == "zlepki": continue
        mask = cv2.imread(mapim)[:,:,0]
        # plt.figure()
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        if drawcontour is None:
            mask = cv2.erode(mask, kernel=np.ones((25, 25)))
        # plt.figure()
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        _, contours = mask2contours(mask)
        
        if drawcontour is None and color is not None:
            for c in contours:
                # calculate moments for each contour
                M = cv2.moments(c)
                if M["m00"]==0: M["m00"]=0.00001
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                 
                
                 # calculate moments for each contour
                M = cv2.moments(c)
                if M["m00"]==0: M["m00"]=0.00001
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                
                
                cY = int(M["m01"] / M["m00"])
                cv2.circle(roi, (cX, cY), dotsize, color, -1)
                cv2.circle(only_points, (cX, cY), dotsize, color, -1)
            
        if drawcontour is not None and color is not None:
            cv2.drawContours(roi, contours, -1, color,9)
            cv2.drawContours(only_points, contours, -1, color,9)
            pass

    return [roi, only_points]


def getclasspoint(namenum, roi, foldermap, dotsize=20, drawcontour=None):
    #if contour == None draws a dot, else a countour
    
    only_points = np.zeros_like(roi) # for an image with only points
    name = namenum[:-1]
    num = namenum[-1:]
    
    # color_dict  = {
    #          # 'parabasal':   (235,249,17),
    #          'HSIL':        (2,7,164),
    #          'ASCH':        (5, 167, 247), #105, 3, 143 5, 167, 247
    #          'LSIL':        (234,104,223),
    #          'ASCUS':       (237, 157, 38),
    #          # 'metaplastic': (135,141,136),
    #          'superficial': (12,255,19)} # incl intermediate

    color_dict  = {
        "ASC-H" : (0,165,255),
        "ASC-US": (180,105,255),
        "HSIL": (0,0,255),
        "LSIL": (255,0,255),
        "NILM": (0,255,0),
        # "Artifacts": (0,255,255)
        }
    ## wczytanie map i wyciągnięcie centrów
    mapimgs = [i for i in glob.glob(os.path.join(foldermap,name+'*')) if i[-5]==num]
    # print(name)
    
    for mapim in mapimgs:
        mapname = os.path.basename(mapim)
        classtype = mapname[21:-5]
        # print(classtype)
           # find color
           # print(classtype.lower())
        if classtype.lower() == "powierzchowne" or classtype.lower() == "powierzchniowe" or classtype.lower() == "posrednie" or classtype.lower() == "pośrednie":
            color = color_dict['NILM']
        elif classtype.lower() == "przypodstawne":
            color = color_dict['NILM']
        elif classtype.lower() == "metaplastyczne" or classtype.lower() == "metaplast":
            color = color_dict['NILM']
        elif classtype.lower() == "HSIL".lower():
            color = color_dict['HSIL']
        elif classtype.lower() == "ASC-H".lower():
            color = color_dict['HSIL']
        elif classtype.lower() == "LSIL".lower() or classtype.lower() == "LSIL nie-ko".lower() or classtype.lower() == "koilocyty":
            color = color_dict['LSIL']
        elif classtype.lower() == "ASC-US".lower() or classtype.lower() == "ASC-US".lower():
            color = color_dict['LSIL']
        else: 
            color = None
                # break
           # print(color)
        
        if classtype == "pojedyncze" or classtype == "zlepki": continue
        mask = cv2.imread(mapim)[:,:,0]
        # plt.figure()
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        if drawcontour is None:
            mask = cv2.erode(mask, kernel=np.ones((25, 25)))
        # plt.figure()
        # plt.imshow(mask)
        # plt.axis('off')
        # plt.show()
        _, contours = mask2contours(mask)
        
        if drawcontour is None and color is not None:
            for c in contours:
                # calculate moments for each contour
                M = cv2.moments(c)
                if M["m00"]==0: M["m00"]=0.00001
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                 
                
                
                cv2.circle(roi, (cX, cY), dotsize, color, -1)
                cv2.circle(only_points, (cX, cY), dotsize, color, -1)
            
        if drawcontour is not None and color is not None:
            cv2.drawContours(roi, contours, -1, color,9)
            cv2.drawContours(only_points, contours, -1, color,9)
            pass

    return [roi, only_points]

