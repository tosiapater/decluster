# -*- coding: utf-8 -*-
 # based on https://keras.io/examples/vision/grad_cam/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from vgg import create_model # local
import os


"""Segment nuclei with gradcam"""
#%% The Grad-CAM algorithm

def get_img_array(img_path, size, expand=0):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    if expand!=0:
        array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#%% resize heatmap to to orginali image's size

def resize_heatmap(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    # jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    # jet_colors = jet(np.arange(256))[:, :3]
    # jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    resize_heatmap = keras.utils.array_to_img(np.stack((heatmap,heatmap,heatmap), axis=-1))
    resize_heatmap = resize_heatmap.resize((img.shape[1], img.shape[0]))
    resize_heatmap = keras.utils.img_to_array(resize_heatmap)
    return resize_heatmap, img

    # Superimpose the heatmap on original image
    # superimpo1sed_img = jet_heatmap * alpha + img
    # superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    
    #%% NN settings

img_size = (224,224,3)
batch_size = 7
epochs = 79
dir_path = ... #TODO: set path to single images
preprocess_input = keras.applications.vgg16.preprocess_input

class_names = ('HSIL', 'LSIL',  'NILM')
# class_names = ('NILM', 'SIL')
n_classes = len(class_names)

model = create_model(img_size, n_classes)
model.load_weights(r"weights\VGG16_3_class.hdf5")
last_conv_layer_name = "block5_conv3"
# Remove last layer's softmax
model.layers[-1].activation = None


save_dir =  ... #TODO: set path to save
os.makedirs(save_dir, exist_ok=True)

#%% data generator

def img_and_name_gen(dir_path, batch_size=4):
    while True:
       batchCount  = 0
       inputs = []
       names = []
    
       onlyfiles = [f for f in os.listdir(dir_path) if f.endswith(".png")]
       
       for img_fname in onlyfiles:
           # print(img_fname)
           img_array = preprocess_input(get_img_array(os.path.join(dir_path, img_fname), size=img_size))
           inputs.append(img_array)
           names.append(img_fname)
           
           batchCount += 1
           
           if batchCount >= batch_size:
               #print('Batch!')
               batchCount = 0
               X = np.array(inputs, np.float32)
               ids = np.array(names)
               
               inputs = []
               names = []
               
               yield X, ids
  
gen = img_and_name_gen(dir_path, batch_size)              
for e in range(epochs):

    img_array, names = next(gen)
    
    preds = model.predict(img_array)
    
    
    for b in range(batch_size):
        # print("Predicted:", class_names[np.argmax(preds[b])])
        
        heatmap_hsil = make_gradcam_heatmap(np.expand_dims(img_array[b], axis=0), model, last_conv_layer_name, pred_index=1)
        # plt.matshow(heatmap_hsil)
    
        heatmap_resized,_ = resize_heatmap(os.path.join(dir_path, names[b]), heatmap_hsil)
        # plt.imshow(heatmap_resized.astype('uint8'))
        # plt.axis('off')
        # plt.show()
    
        # in need of binary image uncomment
        # heatmap_piece = heatmap_resized.copy()
        # heatmap_piece = keras.utils.array_to_img((heatmap_piece>(0.5*255))*255)
        
        heatmap_piece = keras.utils.array_to_img(heatmap_resized)
        heatmap_piece.save(os.path.join(save_dir, names[b]))



#%% Data - single image

# img_path = ...

# display(Image(img_path))

# # Prepare image
# img_array = preprocess_input(get_img_array(img_path, size=img_size, expand=1))

# # Print what the top predicted class is
# preds = model.predict(img_array)

# print("Predicted:", class_names[np.argmax(preds)])
# # Generate class activation heatmap
# heatmap_hsil = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0)
# # heatmap_lsil = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=1)
# # heatmap_nlim = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=2)
# # # Display heatmap
# # plt.matshow(heatmap_hsil)

# heatmap_resized, img = resize_heatmap(img_path, heatmap_hsil)
# plt.figure('heatmap')
# plt.imshow(heatmap_resized.astype('uint8'))
# plt.axis('off')
# plt.show()

# heatmap_piece = heatmap_resized.copy()
# heatmap_piece = (heatmap_piece>(0.5*255))*255  # top 1-n prediction
# # plt.imshow(heatmap_piece.astype('uint8'))
# # plt.axis('off')
# # plt.show()


# superimposed_img = heatmap_piece * 0.4 + img
# plt.imshow(superimposed_img.astype('uint8'))
# plt.axis('off')
# plt.show()

# # superimposed_img = keras.utils.array_to_img(superimposed_img)

# # # save_and_display_gra1dcamdcam(img_path, heatmap_hsil)
# # # save_and_display_gradcam(img_path, heatmap_lsil)
# # # save_and_display_gradcam(img_path, heatmap_nlim)











