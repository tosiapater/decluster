import numpy as np
from PIL import Image
from empatches import EMPatches
from pathlib import Path
from keras.models import load_model
from focal_loss import categorical_focal_loss # local
from segmentation_models import get_preprocessing
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from get_annotations import get_annotations

# nuceli localisation pour CRIC

#%% functions

def tile_gen(patches,preprocess_func=None, batch_size=4, newsize = [256,256]):
    # while True:
    batchCount  = 0
    inputs = []
  
    for inx, patch in enumerate(patches):

        img_resize = (Image.fromarray(patch.astype(np.uint8))).resize(newsize)
        if preprocess_func is not None:
            img_resize = preprocess_func(np.array(img_resize))
        inputs.append((img_resize))
        
        batchCount += 1
        
        if batchCount >= batch_size:
            # print('Batch!')
            batchCount = 0
            X = np.array(inputs, np.float32)
            
            inputs = []
            
            yield X
            
        elif inx == len(patches)-1:
            
            # print('Batch!')
            batchCount = 0
            X = np.array(inputs, np.float32)
            
            inputs = []
            
            yield X
            
            
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
            
            
def get_nuclei_loc(image, unet,preprocess_func,*,patchsize=130, overlap=0.5, batch=12):
    """
    patches an image, segments nuclei and merges the image back

    Parameters
    ----------
    image : TYPE
        cv2 image to extract nuclei.
    unet : TYPE
        Unet with nuclei weighs.
    * : TYPE
        DESCRIPTION.
    patchsize : TYPE, optional
        DESCRIPTION. The default is 130.
    overlap : TYPE, optional
        DESCRIPTION. The default is 0.5.
    batch : TYPE, optional
        DESCRIPTION. The default is 12.

    Returns
    -------
    TYPE
        bw image of segmented nuclei.

    """
    emp = EMPatches()

    org_patches, indices = emp.extract_patches(image, patchsize=patchsize, overlap=overlap)
    tile_generator = tile_gen(org_patches,preprocess_func, batch_size=batch)
    
    unet_preds = unet.predict(tile_generator, steps = np.ceil(len(org_patches)/batch), verbose=0)
    resized_list  = []  
    
    for inx, patch in enumerate(unet_preds):
        
        patch_resize = (Image.fromarray(((patch[:,:,0]>0.5)*255).astype('uint8'))).resize([patchsize, patchsize]) 
        patch_resize = np.array(patch_resize)
        resized_list.append(patch_resize)
    return emp.merge_patches(resized_list, indices, mode='max')

#%%body
if __name__ == "__main__":
    
    
    
    seg_nuc = load_model(
        r"weights\Unet_nuc_densenet169_steps_200_epochs_20_lr_0.0001batchsize8.h5",
        custom_objects={'focal_loss': categorical_focal_loss})

    preprocess_unet = get_preprocessing('densenet169')
    

    f1_scores = []
    sen_scores = []
    names = []

    folderpath = Path(...) # TODO: path to CRIC images
    save_folder = Path(...)
    
    
    
    for file_path in folderpath.rglob('*.png'): # uncomment if all
        file_name = file_path.name
    # for file_name in file_list:
    #     file_path = folderpath / file_name
    
        names.append(file_name)
    
        cellimage = cv2.imread(str(file_path))
        cellimage_shape = cellimage.shape
        nucmapbin = get_annotations(file_name, (cellimage_shape[1],cellimage_shape[0]))
        nucmap = get_annotations(file_name, (cellimage_shape[1],cellimage_shape[0]), 11)
        nuclei = get_nuclei_loc(cellimage,  seg_nuc, preprocess_unet, patchsize=90)  # patchsize aprox 2x diameter of a nucelus
        
        kernel = circle_kernel(11)
        nuclei = cv2.morphologyEx(nuclei.astype('uint8'), cv2.MORPH_OPEN, kernel) # delete too small detections
        nuclei = cv2.dilate(nuclei,kernel,iterations = 3)
        
        # plt.figure(file_name)
        stack_img = np.dstack((nuclei, nucmap*255, cellimage[:,:,0])).astype('uint8')
        # plt.imshow(stack_img)
        # plt.axis('off')
        # plt.show()
        cv2.imwrite(str(save_folder / file_name), cv2.cvtColor(stack_img, cv2.COLOR_BGR2RGB))
    
        # nucmapbin = nucmap > 0
        nucleibin = nuclei > 0
        
        tp = nucmapbin == nucleibin  
        
        nucmapbin100 = nucmapbin*100
        flagmap = nucmapbin100 + nucleibin*1
        
        tp = np.sum(flagmap == 101)
        fn = np.sum(flagmap == 100)
        
        contours, _ = cv2.findContours(nucleibin.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        fp = len(contours) - tp
        
        f1 = 2*tp/(2*tp+fp+fn)
        
        f1_scores.append(f1)
        sen = tp/np.sum(nucmapbin)
        sen_scores.append(sen)
    
    with open('results_F1_90_6layers_cric.txt', 'a') as f:    
        for name, f1 in zip(names, f1_scores):
            print(f'{name}: {f1*100:.2f}', file=f)
        print(f'Average: {np.mean(f1_scores)*100:.2f}', file=f)
    
  
    for name, score in zip(names, sen_scores):
        print(f'{name}: {score*100:.2f}')
    print(f'Average: {np.mean(sen_scores)*100:.2f}')
        
    
    
    
    
    
    
    
    
    