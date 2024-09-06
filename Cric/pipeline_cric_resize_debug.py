# -*- coding: utf-8 -*-


"""
sgementing -classifing pipeline with declusterization
to perform watershed segmetation uncomment parts that  the matlab package
modification: calculates sensitivity for separate cells and clusters separatly

"""

import numpy as np
import os
# from matplotlib import pyplot as plt

from keras.models import load_model
import cv2
import json
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input

from segmentation_models import get_preprocessing

import matlab.engine

from watershed_cutout import watershed # local
from vgg import create_model # local
from getclasspoint import getclasspointcric3class, getclasspointcricbin # local
from focal_loss import categorical_focal_loss # local
from segmenthsilempatchesloop import get_nuclei_loc # local

import time




#%% util functions

def circle_kernel(d):
    """
    circle kernel for morphological operations
    d is the odd size of a square kernel matrix and the diameter of the cricle
    returns a dxd matrix
    """
    matrix = np.zeros((d,d), np.uint8) 
        
    # Center coordinates, radius and color of circle 
    center_coordinates = (int(d/2), int(d/2))      
    radius = int(d/2)
    color = 1 
        
    # Line thickness of -1 px = fill in
    thickness = -1
        

    # Draw a circle of red color of thickness -1 px 
    kernel = cv2.circle(matrix, center_coordinates, radius, color, thickness) 
    return kernel


def marker2cutout(marker1, markers): # some params are global if import add params
    """
    cuts out rectangular matrices around  cv2 watershed marker
    
    in: chosen marker id, all watershed markers
    out: the cutout, binary mask of the marker, outline
    """
    marker1bin = ((markers == marker1)*1).astype(np.uint8)

    kernel = circle_kernel(7)
    marker1ero = cv2.dilate(marker1bin,kernel,iterations=3)

    marker1loc = np.where(markers == marker1)
    marker1map = (marker1ero*255).astype(np.uint8)
    
    marker_mask = cv2.resize(marker1bin, (roiorginalsize_y, roiorginalsize_x))
    
    outline, hierarchy = cv2.findContours(marker1map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    markercorners = [max(marker1loc[0]), max(marker1loc[1]),min(marker1loc[0]), min(marker1loc[1])] # [top corner, bottom corner]
    if abs(markercorners[2]-markercorners[0])<5 or abs(markercorners[2]-markercorners[0])<5 :
        markercell = []
        markercell_mask = []
        

    markercorners = np.array(markercorners)
    markercorners_margin = [min(markercorners[0]+margin, roiorginalsize_x),
                                  min(markercorners[1]+margin, roiorginalsize_y),
                                  max(markercorners[2]-margin, 0),
                                  max(markercorners[3]-margin, 0)]
    corners.append(markercorners_margin)

    markercell = roirgb[markercorners_margin[2]:markercorners_margin[0], markercorners_margin[3]:markercorners_margin[1]]
    markercell_mask = marker1bin[markercorners_margin[2]:markercorners_margin[0], markercorners_margin[3]:markercorners_margin[1]]


    
    return markercell, markercell_mask, outline


def image_gen(folderin, preprocess_func, batch_size=10, newsize=(512,512)): # warning: function uses global variables
    
    batchCount  = 0
    inputs = []
    dirlist = os.listdir(folderin)
    dirlistlen = len(dirlist)
    for inx, filename in enumerate(dirlist):
        f = os.path.join(folderin, filename)
        name, ext = os.path.splitext(filename)
        
        if os.path.isfile(f) and (ext == '.bmp' or ext == '.png'):
            name_list.append(name)
            # load image
            roi = cv2.imread(f)
            roi_list.append(roi)
            roi512 = cv2.resize(roi, newsize)
            processedimg = preprocess_func(roi512)

            inputs.append(processedimg)
            
            batchCount += 1
            
            if batchCount >= batch_size:
                # print('Batch!')
                batchCount = 0
                X = np.array(inputs, np.float32)
                
                inputs = []
                
                yield X
                
            elif inx == dirlistlen-1:
                
                # print('Batch!')
                batchCount = 0
                X = np.array(inputs, np.float32)
                
                inputs = []
                
                yield X
            
            
    
#%% segmentation


folderin = ... # TODO: path to CRIC images
folderout = ...
folderann = r"classifications.json"




margin = 10
margin_cluster = 45 # margin used in cutting out clustered cells

treshold_cluster = 200 + 2*margin # if a segmented area is bigger than the threshold it gets the declustering treatment

image_size_unet = (512,512)
image_size_vgg = (224,224,3)
n_classes=2 
optim = Adam(lr=0.001)
class_model = 'VGG16'

class_names = ('NILM', 'SIL-ASC' )
class_color_cluster = ((0,255,0),(0,0,255))
class_color_separate = ((0,100,0),(0,0,100))

# cytoplasm
unet = load_model(
    r"weights\FPN_densenet169_classes3_steps_200_epochs_40_lr_0.001_aug.hdf5",
    custom_objects={'focal_loss': categorical_focal_loss})

# nuceli
unet_nuc = load_model(
    r"weights\Unet_nuc_densenet169_steps_200_epochs_20_lr_0.0001batchsize8.h5",
    custom_objects={'focal_loss': categorical_focal_loss})
    # r"C:\Users\apater\Documents\Cytologie\weights\Unet_nuclei\Unet_densenet169_steps_200_epochs_20_lr_0.0001.h5"

inputs = []

tp_list = []
cp_list = []
sensitivity_list = []
name_list = []
roi_list = []
found_clustered_list = []
found_separate_list = []
tp_clustered_list = []
tp_separate_list = []
fn_list = []
preprocess_unet = get_preprocessing('densenet169') 
        
 # get U-Net predictions       
image_gen_unet = image_gen(folderin, preprocess_unet, batch_size=10, newsize=image_size_unet)                
unet_preds = unet.predict(image_gen_unet, verbose=0)


roiorginalsize_x, roiorginalsize_y,_ = roi_list[0].shape
resizefactor_x, resizefactor_y = roiorginalsize_x/image_size_unet[0], roiorginalsize_y/image_size_unet[1]
#%% load classification model

vgg_model_ft = create_model(image_size_vgg, n_classes, optim, fine_tune=2, classification_model=class_model)

    
vgg_model_ft.load_weights(r"weights\VGG16_2_class_CRIC.hdf5")

#
#%% classification

anndicts = json.load(open(folderann,))

# turn on matlab
# eng = matlab.engine.start_matlab()
start = time.time()


images_num = len(unet_preds)
# unetoutput = map8u
for inx,unetoutput in enumerate(unet_preds):
    
    
    
    # print(f"processing image: {inx} of {images_num}...")
    
    
    roimarked = roi_list[inx].copy() 
    if n_classes==2:
        point_ref,_ = getclasspointcricbin(name_list[inx], np.zeros_like(roimarked), anndicts, dotsize=0) # dotsize=0 - rysuje jeden piksel
        big_point_ref,_ = getclasspointcricbin(name_list[inx], roimarked, anndicts, dotsize=20) # na analizowanym obrazie
    elif n_classes==3 or n_classes==4:
        point_ref,_ = getclasspointcric3class(name_list[inx], np.zeros_like(roimarked), anndicts, dotsize=0) # dotsize=0 - rysuje jeden piksel
        big_point_ref,_ = getclasspointcric3class(name_list[inx], roimarked, anndicts, dotsize=20) # na analizowanym obrazie
    
    
    
    
    # map8u = (unetoutput[:,:,0]*255).astype(np.uint8)
    map8u = ((np.argmax(unetoutput, axis=2)==0)*1).astype(np.uint8)
# separate found areas
    # inputsvgg = []
    inputvgg_nonprocessed =[]
    corners = []
    outlines = []
    with_mask = []
    
    map8u = cv2.resize(map8u, (roiorginalsize_y, roiorginalsize_x))
    markers = watershed(map8u,0, 0.1)
    
    markerlist = np.unique(markers)[np.unique(markers) > 1]
    roirgb = roi_list[inx].copy()
    roirgb = cv2.cvtColor(roirgb, cv2.COLOR_RGB2BGR)
    
    colormask = np.zeros_like(roimarked)
    # roigray = cv2.cvtColor(roirgb, cv2.COLOR_RGB2GRAY)#debug
    # debugstack = np.dstack((roigray,np.zeros_like(roigray), map8u*255))
    # plt.imshow(debugstack)
    # plt.axis('off')
    # plt.show()

    # cut out cells for classification
    for marker1 in markerlist:
        
        markercell, markercell_mask, outline = marker2cutout(marker1, markers)
        inputvgg_nonprocessed.append(markercell)
        # inputsvgg.append(preprocess_input(markercell))
        with_mask.append(markercell_mask)
        outlines.append(outline)
 

    # check sizes: if large image then do the cluster method      
    big_cuts_inx = [i for i, x in enumerate(inputvgg_nonprocessed) if min(x.shape[:2]) >treshold_cluster]
    big_cuts = [inputvgg_nonprocessed[i] for i in big_cuts_inx]
    
    # small cuts : prepare for prediction
    small_cuts_inx = [i for i, x in enumerate(inputvgg_nonprocessed) if min(x.shape[:2]) <=treshold_cluster and min(x.shape[:2]) > 5]
    found_separate = len(small_cuts_inx)
    if found_separate > 0:
        small_cuts = []
        
        for i in small_cuts_inx:
            
            small_tmp = inputvgg_nonprocessed[i]
            small_tmp = cv2.resize(small_tmp, image_size_vgg[:-1])
            small_cuts.append(preprocess_input(small_tmp))
        
        
    
        Xvgg = np.array(small_cuts)
        # debug_xvg = np.array(inputvgg_nonprocessed)
        # DEBUGLIST.append(debug_xvg)
        
        
        # get vgg classification for small cuts
         
        vgg_preds_ft = vgg_model_ft.predict(Xvgg, verbose=0)
        vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)
        vgg_pred_classes_ft_names = [class_names[i] for i in vgg_pred_classes_ft]
        # print(vgg_pred_classes_ft_names)
    
        

        # visualization
        
        
        for ind, vgg_pred_class in enumerate(vgg_pred_classes_ft):
            # roimarked = cv2.rectangle(roimarked, (corners[ind][1],corners[ind][0]), (corners[ind][3],corners[ind][2]), color=class_color[vgg_pred_class], thickness =9)
            for c in outlines[small_cuts_inx[ind]]:
                # marker1outlineorg = [(i*resizefactor).astype(np.uint8) for i in c]
                
                # cres = c * np.array([resizefactor_x, resizefactor_y])
                percent = max(vgg_preds_ft[ind])*100
                # marker1outlineorg = [(i*resizefactor).astype(np.uint8) for i in c]
                u8cres = c.astype(np.int32)
                textpoint = max(u8cres[:,0,0]), max(u8cres[:,0,1])
                cv2.fillConvexPoly(colormask, u8cres, class_color_separate[vgg_pred_class])
                cv2.drawContours(roimarked, u8cres, -1, class_color_separate[vgg_pred_class], 9)
                # cv2.putText(roimarked, f'{percent:.2f}', textpoint, cv2.FONT_HERSHEY_SIMPLEX, 3, class_color_cluster[vgg_pred_class], 9) 

     # get cluster classification ##################################################################

    found_clusterd=0 # do zliczania komórek w klastrach
    if len(big_cuts_inx) > 0:
    
        corners_cluster = []
        inputvgg_nonprocessed_cluster =[]
        outlines_cluster = []
        inputvgg_cluster =[]
        # with_mask_cluster = []
        
         
        nuclei_maps = [get_nuclei_loc(x, unet_nuc, preprocess_unet, patchsize=120) for x in big_cuts]
        
        for cluster_inx, cluster in enumerate(big_cuts):
            # plt.figure(cluster_inx)
            # plt.imshow(cluster)
            # # plt.axis('off')
            # plt.show()
        # cluster_inx = 0
            nuclei_map_tmp = nuclei_maps[cluster_inx]
            outline_mask = with_mask[big_cuts_inx[cluster_inx]]
            img = inputvgg_nonprocessed[big_cuts_inx[cluster_inx]] 
            
            # get only nuclei within the analized region
            inside_mask_muclei = (np.logical_and(nuclei_map_tmp, outline_mask)*1).astype("uint8")
            

            # change to matlab types
            
            # inside_mask_muclei_mat = matlab.int8(inside_mask_muclei.tolist())
            # img_mat = matlab.int16(img.tolist())
            # outline_mask_mat = matlab.int8(outline_mask.tolist())
            
            # matlab_out = eng.test_waterSeeded_split_tos(outline_mask_mat, True, img_mat, inside_mask_muclei_mat) #(bw_img, gradientOnBW, img, img_fgm)
            # cluster_watershed = np.asarray(matlab_out,dtype=np.uint8)
            
            clustercopy = np.copy(cluster)
            contours_nuclei, _ = cv2.findContours(inside_mask_muclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(clustercopy, contours_nuclei, -1, (255,255,255), 3)

            
            big_cluster_corners = corners[big_cuts_inx[cluster_inx]]#[2:]
            big_cluster_corners_min = (big_cluster_corners[3], big_cluster_corners[2])
            big_cluster_corners_max = (big_cluster_corners[1], big_cluster_corners[0])
    
            colortmp = (255,255,255)
            thicnesstmp = 11
            
            roimarked = cv2.rectangle(roimarked,big_cluster_corners_min,big_cluster_corners_max, colortmp, thicnesstmp)

            
            found_clusterd = found_clusterd + len(contours_nuclei)
            
            # find cluster localization in a roi
            
            
            for contour_cluster in contours_nuclei:
                contour_cluster_org = np.zeros_like(contour_cluster)
                contour_cluster_org[:,0,0] = contour_cluster[:,0,0] + big_cluster_corners_min[0]
                contour_cluster_org[:,0,1] = contour_cluster[:,0,1] + big_cluster_corners_min[1]
                
                M = cv2.moments(contour_cluster_org)
                if M["m00"]==0: M["m00"]=0.00001
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                x0,y0 = (cX - margin_cluster,cY - margin_cluster)
                x1,y1 = (cX + margin_cluster,cY + margin_cluster)
                 

                markercornerorginal = [max(contour_cluster_org[:,0,0]),
                                       max(contour_cluster_org[:,0,1]), 
                                       min(contour_cluster_org[:,0,0]), 
                                       min(contour_cluster_org[:,0,1])]
                
                markercornerorginal_margin = [x1, # margin around nuceli
                                              y1,
                                              max(x0, 0),
                                              max(y0, 0)]
                
                markercell = roirgb[markercornerorginal_margin[3]:markercornerorginal_margin[1],
                                    markercornerorginal_margin[2]:markercornerorginal_margin[0]]
                
                corners_cluster.append(markercornerorginal_margin)
                inputvgg_nonprocessed_cluster.append(markercell)
                outlines_cluster.append(contour_cluster_org)
                

                           
                # big_cluster_corners_min_org = (markercornerorginal[2], markercornerorginal[3])
                # big_cluster_corners_max_org = (markercornerorginal[0], markercornerorginal[1])
    
                # colortmp = (255,255,255)
                # thicnesstmp = 11
                # cv2.drawContours(roimarked, contour_cluster_org, -1, (0,0,0), 3)
                
                # roirgb = cv2.rectangle(roirgb,big_cluster_corners_min_org,big_cluster_corners_max_org, colortmp, thicnesstmp)
                # plt.figure(cluster_inx+100000)
                # plt.imshow(markercell)
                # # plt.axis('off')
                # plt.show()
                
            # cv2.drawContours(roirgb, contours_nuclei, -1, (255,255,255), 3)
            # plt.figure(cluster_inx+100000)
            # plt.imshow(markercell)
            # # plt.axis('off')
            # plt.show()
        if  not inputvgg_nonprocessed_cluster:
            continue
        for input_non_proc in inputvgg_nonprocessed_cluster:
            
            input_tmp = cv2.resize(input_non_proc, image_size_vgg[:-1])
            inputvgg_cluster.append(preprocess_input(input_tmp))
    
        Xvgg = np.array(inputvgg_cluster)
        
        
        vgg_preds_ft = vgg_model_ft.predict(Xvgg, verbose=0)
        vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)
        vgg_pred_classes_ft_names = [class_names[i] for i in vgg_pred_classes_ft]
        # print(vgg_pred_classes_ft_names)
        
        # for cl_name, precent in zip(vgg_pred_classes_ft_names, vgg_preds_ft):
            # print(cl_name, max(precent))
    

        # visualization
        # colormask = np.zeros_like(roi)
        # roimarked = roi_list[inx].copy()
        for ind, vgg_pred_class in enumerate(vgg_pred_classes_ft):
            roimarked = cv2.rectangle(roimarked, (corners_cluster[ind][0],corners_cluster[ind][1]), (corners_cluster[ind][2],corners_cluster[ind][3]), color=class_color_cluster[vgg_pred_class], thickness =9)
            c = outlines_cluster[ind]
            percent = max(vgg_preds_ft[ind])*100
            # marker1outlineorg = [(i*resizefactor).astype(np.uint8) for i in c]
            # cres = resizefactor*c
            u8cres = c.astype(np.int32)
            # textpoint = max(u8cres[:,0,0]), max(u8cres[:,0,1])
            # cv2.fillConvexPoly(colormask, u8cres, class_color_cluster[vgg_pred_class])
            cv2.rectangle(colormask, (corners_cluster[ind][0],corners_cluster[ind][1]), (corners_cluster[ind][2],corners_cluster[ind][3]), color=class_color_cluster[vgg_pred_class], thickness =-1)
            # cv2.drawContours(roimarked, u8cres, -1, class_color_cluster[vgg_pred_class], 9)
            # cv2.putText(roimarked, f'{percent:.2f}', textpoint, cv2.FONT_HERSHEY_SIMPLEX, 3, class_color_cluster[vgg_pred_class], 9) 
    



    # show true class info
    

    TPs_per_class_cluster = []
    TPs_per_class_separate = []
    TPs_per_class = []
    FN_per_class = []
    cond_positive_per_class = []
    sensitivity_per_class = []
    for cluster_class, separate_class in zip(class_color_cluster, class_color_separate):
        # cluster
        map_indices_cluster = np.where(np.all(colormask == cluster_class, axis=-1))
        cluster_class_map =  np.zeros(roimarked.shape[:2])
        cluster_class_map[map_indices_cluster]=1
        # single
        map_indices_single = np.where(np.all(colormask == separate_class, axis=-1))
        single_class_map =  np.zeros(roimarked.shape[:2])
        single_class_map[map_indices_single]=1
        # ground truth
        point_indices = np.where(np.all(point_ref == cluster_class, axis=-1))
        single_class_ref_point = np.zeros(roimarked.shape[:2])
        single_class_ref_point[point_indices]=1
        

        # count TP
        
        # single_class_ref_point = single_class_ref_point*100
        
        condition_positive = np.sum(single_class_ref_point)
        cond_positive_per_class.append(condition_positive)
        # cluster
        whereTP_cluster = single_class_ref_point+cluster_class_map
        TP_num_cluster = np.sum(whereTP_cluster==2)
        TPs_per_class_cluster.append(TP_num_cluster)
        # single
        whereTP_single = single_class_ref_point+single_class_map
        TP_num_single = np.sum(whereTP_single==2)  
        TPs_per_class_separate.append(TP_num_single)
        # sum cluster+single
        TPs_per_class.append(TP_num_cluster+TP_num_single)
        FN_per_class.append(condition_positive-TP_num_cluster-TP_num_single)  # w tym niesklasyfikowane
        # sensitivity_per_class.append(TP_num/condition_positive)
    
    # print(class_names)
    # print(TPs_per_class)
    # print(cond_positive_per_class)
    # print(sensitivity_per_class)
    
    tp_list.append(TPs_per_class)
    cp_list.append(cond_positive_per_class)
    fn_list.append(FN_per_class)
    found_separate_list.append(found_separate)
    found_clustered_list.append(found_clusterd)
    tp_clustered_list.append(TPs_per_class_cluster)
    tp_separate_list.append(TPs_per_class_separate)

    # plt.figure(inx)
    # plt.imshow(cv2.cvtColor(roimarked, cv2.COLOR_RGB2BGR))
    # plt.axis('off')
    # plt.show()
    
    savename = os.path.join(folderout, name_list[inx]+".png")
    # cv2.imwrite(savename, roimarked) 
    

# eng.quit() 

end = time.time()
print("TIME")
print(end - start) 
print((end - start)/images_num)
       

tp_arr = np.array(tp_list)
cp_arr = np.array(cp_list)
fn_arr = np.array(fn_list) 
found_separate_arr = np.array(found_separate_list)
found_clustered_arr = np.array(found_clustered_list)
tp_clustered_arr = np.array(tp_clustered_list)
tp_separate_arr = np.array(tp_separate_list)


tp_sum = sum(tp_arr)
cp_sum = sum(cp_arr)
fn_sum = sum(fn_arr)
sensitivity_sum = (tp_sum/cp_sum*100)
found_separate_sum = sum(found_separate_arr)
found_clustered_sum = sum(found_clustered_arr)
tp_clustered_sum = sum(tp_clustered_arr)
tp_separate_sum = sum(tp_separate_arr)

print(class_names)
print(f"TP: {tp_sum}\nCP: {cp_sum}\nSensitivity :{sensitivity_sum}")
print('\a')

print(f"TP clustered: {tp_clustered_sum}\nFound clustered: {found_clustered_sum}")        
print(f"TP separate: {tp_separate_sum}\nFound separate: {found_separate_sum}") 