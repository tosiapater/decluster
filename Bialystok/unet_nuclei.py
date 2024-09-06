# -*- coding: utf-8 -*-
from segmentation_models import Unet, Linknet, FPN, PSPNet
from segmentation_models import get_preprocessing


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.models import load_model
from keras.metrics import MeanIoU

import tensorflow as tf

import random
import albumentations as A

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import cv2
import os

# local
from mask_to_countours import mask2contours
from evaluation import iou, dice
from focal_loss import categorical_focal_loss

"""find nuclei based on gradcam  grayscale results
this code is for both learning and testing. to only test comment the fitting and load the weights
code is optimized for the following data structure: directory -- train/val/test --- image/ image map
"""
#%% parameters
# backbone
BACKBONE = 'densenet169'

#VGG		'vgg16' 'vgg19'
#ResNet		'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
#SE-ResNet	'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
#ResNeXt	'resnext50' 'resnext101'
#SE-ResNeXt	'seresnext50' 'seresnet101'
#SENet154	'senet154'
#DenseNet	'densenet121' 'densenet169' 'densenet201'
#Inception	'inceptionv3' 'inceptionresnetv2'
#MobileNet	'mobilenet' 'mobilenetv2'

# set train parameters
OPT_MODEL = 'Unet' # Linknet, Unet, FPN, PSPNet
OPT_WEIGHTS = 'imagenet'
OPT_CLASS = 3 

OPT_CLASS_WEIGHTS = None 
STEPS = 200
EPOCHS = 20
IMG_SIZE1 = 256
IMG_SIZE2 = IMG_SIZE1
IMG_CH = 3

LR = 0.0001 
LOSS = categorical_focal_loss() #'categorical_crossentropy'
METRICS = ['accuracy']#, MeanIoU(num_classes=OPT_CLASS)] #'accuracy'


datapath = Path(...)  #TODO: set path to images

savepath = ... #TODO: set path for results

#%%  util functions



def preprocess(img,temp_target, aug=1, gradcam_tresh=0.5): 

    temp_target = ((temp_target>(gradcam_tresh*np.max(temp_target)))*255).astype('uint8') # use the treshold to determine how much of the gradcam gradient to use

    
    preprocess_input = get_preprocessing(BACKBONE)
    if aug is not None:
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.5,rotate_limit=90, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, brightness_by_max=True, p=0.4),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=(-20,20), p=0.5),
            A.FancyPCA(alpha=1.2, p=0.1),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=0.3),
            A.Blur(blur_limit=12, p=0.3),
            A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3), # interpolacja?
            A.GaussianBlur(blur_limit=(7,7), sigma_limit=10, p=0.3),
            A.GaussNoise(var_limit=(80.0), mean=0, per_channel=True, p=0.3),
       
            A.Sharpen(alpha=(0.3), lightness=(0.8,1), p=0.3),
            A.ElasticTransform(p=0.7, alpha=224*2, sigma=224 * 0.04, alpha_affine=224*0.03)
            
        ], p=0.9)
      
        transformed = transform(image=img.astype(np.uint8), mask =temp_target.astype(np.uint8))
        processedimg = preprocess_input(transformed['image'])
        target_mask = transformed['mask']
    else:
        processedimg = preprocess_input(img.astype(np.uint8))
        target_mask = temp_target
        
    if OPT_CLASS == 1 :
        return(processedimg, target_mask/255)
    else:
        temp_target_contour, _ = mask2contours(target_mask)
        target_mask = (np.round(target_mask/255)).astype(np.uint8)
        temp_target_thick = cv2.subtract(target_mask, temp_target_contour)
        
        temp_target_stack = np.stack((temp_target_thick,temp_target_contour), axis=-1).astype('float')
        background = 1 - temp_target_stack.sum(axis=-1, keepdims=True)
        temp_target_stack = np.concatenate((temp_target_stack, background), axis=-1)
            
        return(processedimg, temp_target_stack)

# def masks_with_outlines(temp_target):
#         # print(temp_target.shape)
#         temp_target = (temp_target[:,:,0]).astype(np.uint8)
#         # print(temp_target.shape)
#         # print(np.max(temp_target))
#         temp_target_contour, _ = mask2contours(temp_target)
#         temp_target = (np.round(temp_target/255)).astype(np.uint8)
#         temp_target_thick = cv2.subtract(temp_target, temp_target_contour)
        
#         temp_target_stack = np.stack((temp_target_thick,temp_target_contour), axis=-1).astype('float')
#         background = 1 - temp_target_stack.sum(axis=-1, keepdims=True)
#         temp_target_stack = np.concatenate((temp_target_stack, background), axis=-1)
#         return temp_target_stack.astype(np.uint8)


def print_learning(myhist):
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(myhist['accuracy'])
    plt.plot(myhist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(2)
    plt.plot(myhist['loss'])
    plt.plot(myhist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    # summarize history for time
    # plt.figure(3)
    # plt.plot(myhist['times'])
    # #plt.plot(myhist['val_loss'])
    # plt.title('model time')
    # plt.ylabel('time')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    
def add_sample_weights(label):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    
    class_weights = OPT_CLASS_WEIGHTS
    class_weights = class_weights/np.sum(class_weights)
    tempweight = np.zeros_like(label)

    # print(label.shape[2])
    for dim in range(label.shape[2]):
        tempweight[:,:,dim] = label[:,:,dim]* class_weights[dim]
        
    # sample_weights = np.sum(tempweight, axis=-1)
    sample_weights = tempweight
    
    # plt.figure()
    # plt.imshow(sample_weights)
    # plt.axis('off')
    # plt.show()
    
    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    # sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    
    return  sample_weights


#%% custom generator

def lr_generator(img_folder_path, f_input = "images", f_target = "grayscale",
                 image_size = (512,512), batch_size = 8,shuffle = False, gradcam_tresh = 0.5, # cutout for the gradcam "probalilites"
                 opt_color = 'rgb', aug = 1, onTesting=False, do_preprocess=1):


    
 while True:
    batchCount  = 0
    inputs = []
    targets = []

    
    
    mypath_image = os.path.join(img_folder_path,f_input)
    mypath_label = os.path.join(img_folder_path,f_target)
    
    onlyfiles = [f for f in os.listdir(mypath_image) if os.path.isfile(os.path.join(mypath_image, f))]
    
    if shuffle == True:
        random.shuffle(onlyfiles)
    
    for img_fname in onlyfiles:
        
        
        if (opt_color == 'gray'):
            temp_image = cv2.imread(os.path.join(mypath_image, img_fname), cv2.IMREAD_GRAYSCALE) 
        elif (opt_color == 'rgb'):
            temp_image = cv2.imread(os.path.join(mypath_image, img_fname), cv2.IMREAD_COLOR) 
        else:
            temp_image = cv2.imread(os.path.join(mypath_image, img_fname), cv2.IMREAD_UNCHANGED) 
            
        temp_target = cv2.imread(os.path.join(mypath_label, img_fname), cv2.IMREAD_GRAYSCALE) 
        
        if temp_target is None:
            continue
        
        if temp_image is None:
            print(str(img_fname)+' no such file!')
            #tmp_idx += 1
            continue
        
        
        temp_image = cv2.resize(temp_image, image_size)
        temp_target = (cv2.resize(temp_target, image_size))
        
        #print("after resize "+str(np.array(temp_image2).shape))
        
        if do_preprocess ==1 :

            temp_image_processed, temp_target_processed = preprocess(temp_image, temp_target, aug)
            inputs.append(temp_image_processed)
            targets.append(temp_target_processed)
        else:
            inputs.append(temp_image)
            targets.append(temp_target)
        
        batchCount += 1
        #print(str(batchCount))
        
        if batchCount >= batch_size:
            #print('Batch!')
            batchCount = 0
            X = np.array(inputs, np.float32)
            y = np.array(targets, np.float32)
            # w = np.array(weights)
            
            #print(str(X.shape))
            
            if (opt_color == 'gray'): X = np.expand_dims(X, axis=3)
            
            #X = np.expand_dims(X, axis=3)
            # y = np.expand_dims(y, axis=3)
            
            #print(str(X.shape))
            
            inputs = []
            targets = []
            
            if (onTesting == True):
                yield X
            else:
                # yield X, y, w
                yield X, y

train_generator = lr_generator(datapath/"train", aug=1, image_size=(IMG_SIZE1,IMG_SIZE2))#,shuffle = True)
val_generator = lr_generator(datapath/"val", aug=None, image_size=(IMG_SIZE1,IMG_SIZE2))#,shuffle = True)

#%% model
# define model
if(OPT_MODEL=='Unet'):
    model  = Unet(backbone_name=BACKBONE, classes = OPT_CLASS,  encoder_weights=OPT_WEIGHTS, #encoder_freeze=True, 
                  input_shape=(IMG_SIZE1, IMG_SIZE2,3), activation='softmax')
elif(OPT_MODEL=='Linknet'):
    model  = Linknet(backbone_name=BACKBONE, encoder_weights=OPT_WEIGHTS)
elif(OPT_MODEL=='FPN'):
    model  = FPN(backbone_name=BACKBONE, classes=OPT_CLASS, encoder_weights=OPT_WEIGHTS, input_shape=(IMG_SIZE1, IMG_SIZE2, IMG_CH), activation='softmax')
elif(OPT_MODEL=='PSPNet'):
    model  = PSPNet(backbone_name=BACKBONE,classes=OPT_CLASS, encoder_weights=OPT_WEIGHTS, input_shape=(IMG_SIZE1, IMG_SIZE2, IMG_CH), activation='softmax')
adamopt = Adam(learning_rate=LR, decay=LR/10) # SGD(learning_rate=LR) #Adam(learning_rate=LR, decay=LR/100)
model.compile(optimizer = adamopt, loss = LOSS, metrics = METRICS)
# model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])


model_string = OPT_MODEL+"_"+BACKBONE+"_classes"+str(OPT_CLASS)+"_steps_"+str(STEPS)+"_epochs_"+str(EPOCHS)+"_lr_"+str(LR)
print(model_string)

model_checkpoint = ModelCheckpoint(('Decluster\\weights\\'+model_string+'_gradcam_nuclei_tl_90_bin.hdf5'), monitor='loss',verbose=1, save_best_only=True)


#%% fit model 
#TODO if testing comment

hist = model.fit(train_generator, steps_per_epoch=STEPS, epochs=EPOCHS, class_weight=OPT_CLASS_WEIGHTS,
                  validation_data=val_generator, validation_steps=121, callbacks=[model_checkpoint])

print_learning(hist.history)


#%% results

#TODO if only testing uncomment
# model = load_model(r"weights\Unet_nuc_densenet169_steps_200_epochs_20_lr_0.0001batchsize8.hdf5", compile=False)

test_generator = lr_generator(datapath/"test", aug=None, image_size=(IMG_SIZE1,IMG_SIZE2), onTesting=True)
my_preds = model.predict(test_generator, steps = 20, verbose=1)

test_im_gen = lr_generator(datapath/"test", batch_size=80, image_size=(IMG_SIZE1,IMG_SIZE2), aug=None, do_preprocess=0)
test_imgs,test_masks = next(test_im_gen)
#%%

ious = []
dices = []
bigious = []
bigdices = []
for ind,temp_result in enumerate(my_preds):
    
    test_mask= (test_masks[ind,:,:]>(255*0.7))*255 # no outiles included
    if np.sum(test_mask)==0: continue
    
    
    output_mask = (temp_result[:,:,0] >0.5)*255
    # output_mask = (temp_result>0.5)*255
    # output_mask = (np.argmax(temp_result, axis=2)>0)*255
    
    ious.append(iou(test_mask, output_mask))
    dices.append(dice(test_mask, output_mask))
    
    
    mask_im = (np.stack((np.zeros_like(output_mask),test_mask ,output_mask), axis=2)).astype('uint8')
    plt.figure()
    plt.imshow((temp_result*255).astype('uint8'))
    plt.axis('off')
    plt.show()
    
    temp_img = cv2.cvtColor(test_imgs[ind,:,:,:].astype('uint8'), cv2.COLOR_BGR2GRAY) 
    temp_img = (np.stack((temp_img, temp_img,temp_img), axis=2))
    # temp_img = test_imgs[ind,:,:,:].astype('uint8')
    superimposed_img = cv2.addWeighted(temp_img,0.5,mask_im,0.5,0)
    
    plt.figure()
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
    
    cv2.imwrite(savepath+'\im'+str(ind)+'.png', superimposed_img) 
    
print("no outline iou")
print(np.mean(ious))
print("no outline dice")
print(np.mean(dices))

print("with outline iou")
print(np.mean(bigious))
print("with outline dice")
print(np.mean(bigdices))





# for image in t[1]:
#     plt.figure()
#     plt.imshow(image)
#     plt.axis('off')
#     plt.show()