#!/usr/bin/env python
# coding: utf-8

#%% imports

######################################### use env cuda11_clone !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import os
from tensorflow import where, zeros_like
from tensorflow.math import is_nan
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import Recall, Precision#, CategoricalAccuracy
import keras.backend as K
from pathlib import Path
import numpy as np
# from livelossplot.inputs.keras import PlotLossesCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, recall_score
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix #, precision_recall_fscore_support

# from focal_loss import categorical_focal_loss
from evaluation import roc_all, roc_binary

# https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/


#%% functions


def preprocess(img): 
    
    transform = A.Compose([
        
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, brightness_by_max=True, p=0.4),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=(-20,20), p=0.5),
        A.FancyPCA(alpha=1.2, p=0.1),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8,8), p=0.3),
        A.Blur(blur_limit=12, p=0.3),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3), # random crop o 10%, flip
        A.GaussianBlur(blur_limit=(7,7), sigma_limit=10, p=0.3),
        A.GaussNoise(var_limit=(80.0), mean=0, per_channel=True, p=0.3),
        A.Sharpen(alpha=(0.3), lightness=(0.8,1), p=0.3),
        A.ElasticTransform(p=0.7, alpha=224*2, sigma=224 * 0.04, alpha_affine=224*0.03)
        
    ], p=0.9)
  
    transformed = transform(image=img.astype(np.uint8))["image"]
    processedimg = preprocess_input(transformed)
    return(processedimg)
    
def f1_macro(y_true, y_pred): # taken from https://www.kaggle.com/code/guglielmocamporese/macro-f1-score-keras/notebook
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = where(is_nan(f1), zeros_like(f1), f1)
    return K.mean(f1)


def create_model(input_shape, n_classes, optimizer=Adam(), fine_tune=1, classification_model='VGG16'):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    if classification_model == 'VGG16':
        conv_base = VGG16(include_top=False,
                         weights='imagenet', 
                         input_shape=input_shape)
      
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    # top_model = Dropout(0.2)(top_model)
    # top_model = Dense(2024, activation='relu')(top_model)
    # top_model = Dropout(0.2)(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)


    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss=CategoricalCrossentropy(), #'binary_crossentropy',
                  metrics=[f1_macro, Recall(), Precision()])# CategoricalAccuracy())# 'accuracy')
    
    # for i, layer in enumerate(vgg_model.layers):
    #     print(i, layer.name, layer.trainable)
    
    return model



#%% data and model
if __name__ == "__main__":
    
    BATCH_SIZE = 64
    input_shape = (224, 224, 3) #(224, 224, 3)
    n_epochs = 40
    learining_rate = 0.001 #TODO set params
    
    patch_size = 90
    
    classification_model = 'VGG16' # VGG16, VGG19, InceptionResNetV2, ConvNeXtLarge, EfficientNetB7, NASNetLarge, DenseNet201, DenseNet121


# TODO set nr of layers
    num_fine_tune_layers = 12 # num of layers to finetune (from the end)
    
    savedir = r'weights\weights'+classification_model+'_3_class_cric_tl_sccin' + str([patch_size]) +'x'+str([patch_size]) +'_epochs{epoch:04d}_lr'+str(learining_rate)+'laters_to_finetune'+str(num_fine_tune_layers)+'.hdf5'
    download_dir = Path(r"CRIC\Cropped\\" + str(patch_size))
    
    train_generator = ImageDataGenerator(#rotation_range=90, 
                                         # width_shift_range=0.5, # czy padding czy rozszezanie
                                         # height_shift_range=0.5,
                                         horizontal_flip=True, 
                                         vertical_flip=True,
                                         # validation_split=0.2, #TODO  only if training on cells 3 class new
                                         preprocessing_function=preprocess) # VGG16 preprocessing     
    
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)#, validation_split=0.5) # VGG16 preprocessing

    
    train_data_dir = download_dir/'train'
    test_data_dir =  download_dir/'test'
    val_data_dir = download_dir/'val'
    
    class_subset = sorted(os.listdir(test_data_dir))
    
    traingen = train_generator.flow_from_directory(train_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   # subset='training',
                                                   batch_size=BATCH_SIZE, 
                                                   shuffle=True)
    
    validgen = test_generator.flow_from_directory(val_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   # subset='training',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)
    
    testgen = test_generator.flow_from_directory(test_data_dir,
                                                 target_size=(224, 224),
                                                 class_mode=None,
                                                 classes=class_subset,
                                                 # subset='validation',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)
    
    n_steps = traingen.samples // BATCH_SIZE
    # n_steps = 200
    n_val_steps = validgen.samples // BATCH_SIZE
    
    
    
    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath=savedir,
                                      monitor="val_f1_macro",
                                      save_weights_only=True,
                                      save_freq=20*BATCH_SIZE,
                                      verbose=1)
    
    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_f1_macro',
                               patience=80,
                               restore_best_weights=True,
                               mode='max')
    
    
    
    true_classes = testgen.classes
        
    lr_schedule = ExponentialDecay(
        learining_rate, decay_steps=2*n_steps, decay_rate=0.98)
    # optim = SGD(lr_schedule)
    optim = SGD(learining_rate)
    # fine-tune - num of layers to unfreze
    vgg_model = create_model(input_shape, len(class_subset), optim, fine_tune=num_fine_tune_layers, classification_model=classification_model)
    # vgg_model.load_weights(r"C:\Users\apater\Documents\Cytologie\weights\VGG16_4_class\epochs0166_lr0.001_4_class.hdf5") # TODO usunąć jesli nie trenuje z outputami uneta
    
    # %% Retrain model with fine-tuning
    
    vgg_ft_history = vgg_model.fit(traingen,
                                        epochs=n_epochs,
                                        validation_data=validgen,
                                        steps_per_epoch=n_steps, 
                                        validation_steps=n_val_steps,
                                        callbacks=[tl_checkpoint_1], #, early_stop],
                                        verbose=1)
    
      #%%

    plt.plot(vgg_ft_history.history['val_f1_macro'], label='val_f1_macro')
    plt.plot(vgg_ft_history.history['f1_macro'], label='f1_macro')
    plt.legend()
    plt.show()
    plt.plot(vgg_ft_history.history['val_loss'], label='val_loss')
    plt.plot(vgg_ft_history.history['loss'], label='loss')
    plt.legend()
    #%% Generate predictions

    vgg_preds_ft = vgg_model.predict(testgen)
    vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)

    vgg_acc_score = accuracy_score(true_classes, vgg_pred_classes_ft)
    print(classification_model+"  Model accuracy: {:.2f}%".format(vgg_acc_score * 100))    
    vgg_f1_score = f1_score(true_classes, vgg_pred_classes_ft, average="macro")
    print(classification_model+" Model F1: {:.2f}%".format(vgg_f1_score * 100))
    vgg_precision_score = precision_score(true_classes, vgg_pred_classes_ft, average="macro")
    print(classification_model+"  Model precision: {:.2f}%".format(vgg_precision_score * 100))
    vgg_recall_score = recall_score(true_classes, vgg_pred_classes_ft, average="macro")
    print(classification_model+"  Model recall: {:.2f}%".format(vgg_recall_score * 100))
    
    #%% heatmap
    
    # Get the names of the  classes
    class_names = testgen.class_indices.keys()
    
    def plot_heatmap(y_true, y_pred, class_names, ax, title):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            square=True, 
            xticklabels=list(class_names), 
            yticklabels=list(class_names),
            fmt='d', 
            cmap=plt.cm.Greens,
            cbar=False,
            ax=ax
        )
        ax.set_title(title, fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    fig, (ax1) = plt.subplots(1, 1)
    
      
    plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax1, title=" ")    
    
    fig.suptitle(" ", fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(top=1.25)
    plt.show()
    
    clreport = classification_report(true_classes, vgg_pred_classes_ft, target_names=list(class_names), digits=4)
    print(clreport)
    #%%roc display
    roc_all(true_classes, vgg_preds_ft, list(class_names))
    
    # print('\a')

    # roc_binary(true_classes, vgg_preds_ft[:,1])
# 
   




