# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
from empatches import EMPatches


"""
 divides images into patches, segments nuclei and stiches back
 
"""

#%% tile generator + summary function

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


    
    
