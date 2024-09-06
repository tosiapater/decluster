import os
import cv2
import numpy as np


    
        
def watershed(mapim,treshold, threshtram):
    
    
    ret, thresh = cv2.threshold(mapim, treshold, 255, cv2.THRESH_OTSU)
    
    # watershed
    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 21)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=21)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,3) # DIST_L1 = |x1-x2| + |y1-y2|
    ret, sure_fg = cv2.threshold(dist_transform,threshtram*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    # fig = plt.figure(2)
    # plt.imshow(sure_fg)
    # plt.axis('off')
    # plt.show()
    
    markers = cv2.watershed(cv2.cvtColor(mapim, cv2.COLOR_GRAY2RGB),markers)
    
    return markers

def cutoutcell(mark, markers, background):
    singlemap = np.uint8(np.zeros_like(markers))
    singlemap[markers==mark] = 1
    kernel = np.ones((3,3), np.uint8)
    singlemap = cv2.dilate(singlemap, kernel, iterations=5)
    ret, thresh = cv2.threshold(singlemap,0.9, 1 ,cv2.THRESH_BINARY)
    conimg, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contours_poly = cv2.approxPolyDP(contours[0], 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        piece = imagergb640[int(boundRect[1]):int(boundRect[1]+boundRect[3]), int(boundRect[0]):int(boundRect[0]+boundRect[2]),:]
        return (piece, boundRect)
    else: return (None,None)
        
    
