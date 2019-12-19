#Importer tous les dépendences et les bibliothéque qu'on aura besoin
#Importer les dependences
import os
import functools
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (15,15)
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

import cv2


from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 
import tensorflow_addons as tfa

#Copy the dataset_folder & convertcsv
#les images ont d'abord été segmentées manuellement à l'aide de l'outil VGG Image Annotator qui produit le fichier csv "convertcsv.csv"
#contenant les coordonnées des différents polygones de segmentation, à l'aide de ce fichier, un masque a été créé pour chaque image
!cp -r /content/drive/My\ Drive/Data/res_img /content/squared_dataset
!cp -r /content/drive/My\ Drive/Data/mask_img /content/squared_mask

!cp /content/drive/My\ Drive/Data/convertcsv.csv /content/convertcsv.csv

!mkdir models
!ls squared_dataset
!ls squared_mask
!ls models

#Lire le fichier csv, Preparer le chemin des images et des masks, et Deviser la dataset en deux parties (training et testing)
regions_file ='convertcsv.csv'
img_folder ='squared_dataset'
mask_folder ='squared_mask'
regions_data = pd.read_csv(regions_file)

img_names =regions_data['#filename'].map(lambda s: os.path.join(img_folder, s))
img_names =list(collections.Counter(img_names).keys())

mask_names =regions_data['#filename'].map(lambda s: os.path.join(mask_folder, s.split('.')[0]+'_mask.JPG'))
mask_names =list(collections.Counter(mask_names).keys())
print('len(img_names): {} \nlen(mask_names): {} \nimg_names[:5]: {} \nmask_names[:5]: {}'.format(len(img_names), len(mask_names), img_names[:5], mask_names[:5]))

tr_img_names, ts_img_names, tr_mask_names, ts_mask_names =train_test_split(img_names, mask_names, test_size=.2, random_state=0)
print('\nlen(tr_img_names): {} \nlen(ts_img_names): {} \ntr_img_names[:5]: {} \ntr_mask_names[:5]: {} \nts_image_names[:5]: {} \nts_mask_names[:5]: {}'.format(len(tr_img_names), len(ts_img_names), tr_img_names[:5], tr_mask_names[:5], ts_img_names[:5], ts_mask_names[:5]))


#Fonction qui nous permet de visualiser l'image et son mask
def show_img_mask(img_file):
  mask_file =os.path.join(mask_folder, img_file.split('/')[1].split('.')[0]+'_mask.JPG')
  print(mask_file)

  img =cv2.imread(img_file)
  img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  mask =cv2.imread(mask_file)
  mask =cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
  blend =cv2.addWeighted(img, .7, mask, .3, .0)
  
  plt.figure(figsize=(20, 20))
  plt.subplot(1, 3, 1)
  plt.imshow(img)
  plt.subplot(1, 3, 2)
  plt.imshow(mask)
  plt.subplot(1, 3, 3)
  plt.imshow(blend)
  
show_img_mask('squared_dataset/obama_0.jpg')
show_img_mask('squared_dataset/trump_249.jpg')

#Créer la base de donner et utiliser la data augmentation pour améliorer la précision du modèle
#Les paramétres d'entrainement
img_shape =(256, 256, 3)
batch_size =3
epochs =30

#les fonctions utilisées pour Data augmentation
scale = 1/255.
def read_img_mask(img_path, mask_path):
    
    img_bf =tf.io.read_file(img_path)
    img =tf.image.decode_jpeg(img_bf, channels =3)

    mask_bf =tf.io.read_file(mask_path)
    mask =tf.image.decode_jpeg(mask_bf, channels=3)

    return img, mask

def shift_img_mask(img, mask, bl,width_shift_range, height_shift_range):
    if bl :
      if width_shift_range :
        width_shift_range =tf.keras.backend.random_uniform([], -width_shift_range*img_shape[1], width_shift_range*img_shape[1])
      if height_shift_range:
        height_shift_range =tf.keras.backend.random_uniform([], -height_shift_range*img_shape[0], height_shift_range*img_shape[0])
      img =tfa.image.translate(img, [width_shift_range, height_shift_range])
      mask =tfa.image.translate(mask, [width_shift_range, height_shift_range])

    return img, mask
      
def flip_img_mask(img, mask):

    flip_prob =tf.keras.backend.random_uniform([], 0.0, 1.0)
    img, mask = tf.cond(tf.less(flip_prob, .5), lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(mask)), lambda: (img, mask))

    return img, mask



def augment(img, mask,bl,hue_delta,resize,width_shift_range, height_shift_range):
    img =tf.image.resize(img, resize)
    mask =tf.image.resize(mask, resize)
    if hue_delta:
      img =tf.image.random_hue(img, hue_delta)
    img, mask =flip_img_mask(img, mask)
    img, mask =shift_img_mask(img, mask, bl,width_shift_range, height_shift_range)
    img =tf.compat.v1.to_float(img)*scale
    mask =tf.compat.v1.to_float(mask)*scale
    return img, mask

def get_dataset(img_names, mask_names, preproc_fn=functools.partial(augment), batch_size=batch_size, shuffle=True):
  
  num_img =len(img_names)
  dataset =tf.data.Dataset.from_tensor_slices((img_names, mask_names))
  dataset =dataset.map(read_img_mask, num_parallel_calls=5)
  dataset =dataset.map(preproc_fn, num_parallel_calls=5)
  dataset =dataset.shuffle(num_img)
  dataset =dataset.repeat().batch(batch_size)
  
  return dataset

tr_cfg ={'width_shift_range': 0.1, 'height_shift_range': 0.1,'resize': [img_shape[0], img_shape[1]],'bl': 1.,'hue_delta': 0.1,}
tr_preproc_fn =functools.partial(augment, **tr_cfg)

ts_cfg ={'width_shift_range': 0.0, 'height_shift_range':0.0, 'resize': [img_shape[0], img_shape[1]],'bl': 0.,'hue_delta': 0.}
ts_preproc_fn =functools.partial(augment, **ts_cfg)

#créer la dataset d'entrainement et la dataset de test
tr_dataset =get_dataset(tr_img_names, tr_mask_names, preproc_fn=tr_preproc_fn)
ts_dataset =get_dataset(ts_img_names, ts_mask_names, preproc_fn=ts_preproc_fn)

print(tr_dataset, '\n', ts_dataset)

#Visualiser l'effet de l'augmentation
with tf.compat.v1.Session() as sess:
  img, mask =sess.run(read_img_mask('squared_dataset/obama_0.jpg', 'squared_mask/obama_0_mask.JPG'))
  img1, mask1 =sess.run(augment(img, mask,bl=1, hue_delta=.1,resize=[img_shape[0], img_shape[1]], width_shift_range=.1, height_shift_range=.1))
  img2, mask2 =sess.run(augment(img, mask,bl=1, hue_delta=.1,resize=[img_shape[0], img_shape[1]], width_shift_range=.1, height_shift_range=.1))

  plt.subplot(2, 3, 1)
  plt.imshow(img)
  plt.subplot(2, 3, 4)
  plt.imshow(mask)
  
  plt.subplot(2, 3, 2)
  plt.imshow(img1)
  plt.subplot(2, 3, 5)
  plt.imshow(mask1)
  
  plt.subplot(2, 3, 3)
  plt.imshow(img2)
  plt.subplot(2, 3, 6)
  plt.imshow(mask2)


#Créer le modèle
def conv_block(input_tensor, num_filters):
  encoder =layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder =layers.BatchNormalization()(encoder)
  encoder =layers.Activation('relu')(encoder)
  encoder =layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder =layers.BatchNormalization()(encoder)
  encoder =layers.Activation('relu')(encoder)
  
  return encoder 

def encoder_block(input_tensor, num_filters):
  encoder =layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder =layers.BatchNormalization()(encoder)
  encoder =layers.Activation('relu')(encoder)
  encoder =layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder =layers.BatchNormalization()(encoder)
  encoder =layers.Activation('relu')(encoder)
  encoder_pool =layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder, encoder_pool

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder =layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), 
                                  padding='same')(input_tensor)
  decoder =layers.concatenate([decoder, concat_tensor], axis=-1)
  decoder =layers.BatchNormalization()(decoder)
  decoder =layers.Activation('relu')(decoder)
  decoder =layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder =layers.BatchNormalization()(decoder)
  decoder =layers.Activation('relu')(decoder)
  decoder =layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder =layers.BatchNormalization()(decoder)
  decoder =layers.Activation('relu')(decoder)
  
  return decoder


inputs =layers.Input(shape=img_shape) #256

encoder0, encoder0_pool =encoder_block(inputs, 32) #128
encoder1, encoder1_pool =encoder_block(encoder0_pool, 64) #64
encoder2, encoder2_pool =encoder_block(encoder1_pool, 128) #32
encoder3, encoder3_pool =encoder_block(encoder2_pool, 256) #16
encoder4, encoder4_pool =encoder_block(encoder3_pool, 512) #8

center =conv_block(encoder4_pool, 1024) #8

decoder4 =decoder_block(center, encoder4, 512) #16
decoder3 =decoder_block(decoder4, encoder3, 256) #32
decoder2 =decoder_block(decoder3, encoder2, 128) #64
decoder1 =decoder_block(decoder2, encoder1, 64) #128
decoder0 =decoder_block(decoder1, encoder0, 32) #256

outputs =layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0) #activation to be determined maybe use softmax

model =models.Model(inputs=[inputs], outputs=[outputs])

model =models.Sequential()
model.add(layers.Flatten(input_shape=img_shape))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(196608, activation='sigmoid'))
model.add(layers.Reshape(img_shape))

def dice_coeff(y_true, y_pred):
  smooth = 1.
  y_true_f =tf.cast(tf.reshape(y_true, [-1]), tf.float32)
  y_pred_f =tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
  
  return score

def mse_loss(y_true, y_pred):
  y_true =tf.cast(y_true, tf.float32)
  y_pred =tf.cast(y_pred, tf.float32)
  loss =tf.reduce_mean(tf.pow(tf.reshape(y_true, shape=[-1, 1]) - tf.reshape(y_pred, shape=[-1, 1]), 2)) * 100
  
  return loss


def ce_loss(y_true, y_pred):
  y_true_f =tf.cast(tf.reshape(y_true, [-1]), tf.float32)
  y_pred_f =tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
  loss = -tf.reduce_sum(y_true_f*tf.log(tf.clip_by_value(y_pred_f,1e-10,1.0)))
  
  return loss

def ce_dice_loss(y_true, y_pred):
  loss =ce_loss(y_true, y_pred) +1-dice_coeff(y_true, y_pred)
  
  return loss

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss



model.compile(optimizer='Adam', loss=mse_loss, metrics=[dice_coeff])
model.summary()

save_model_path = 'models/model_2.hdf5'
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

epochs = 300
history = model.fit(tr_dataset, 
                   steps_per_epoch=int(np.ceil(len(tr_img_names) / float(batch_size))), 
                   epochs=epochs, 
                   validation_data=ts_dataset, 
                   validation_steps=int(np.ceil(len(ts_img_names) / float(batch_size))), 
                   callbacks=[cp])
