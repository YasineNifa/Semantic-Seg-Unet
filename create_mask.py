
#Importer les dependences
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2




regions_file ='convertcsv.csv'
img_folder ='squared_dataset'
mask_folder ='squared_mask'

regions_data = pd.read_csv(regions_file)


def create_mask(mask_file, pts, color_):
  img =cv2.imread(img_file)
  mask =np.zeros(img.shape, dtype = "uint8")
  cv2.fillPoly(mask, [pts], color=color_)
  cv2.imwrite(mask_file, mask)
  
def add_region(mask_file, pts, color_):
  mask =cv2.imread(mask_file)
  cv2.fillPoly(mask, [pts], color=color_)
  cv2.imwrite(mask_file, mask)



blue =(255, 0, 0) #obama
green =(0, 255, 0) #trump

for indx, row in regions_data.iterrows():
  dict_rsa =eval(row['region_shape_attributes'])
  pts =np.array(list(zip(dict_rsa['all_points_x'], dict_rsa['all_points_y'])))
  img_file =os.path.join(img_folder, row['#filename'])
  mask_file =os.path.join(mask_folder, row['#filename'].split('.')[0]+'_mask.JPG')
  
  if row['region_id'] ==0:
    color =blue
  elif row['region_id'] ==1:
    color =green
  
  if os.path.isfile(mask_file):
    print('open {} and add region'.format(mask_file))
    add_region(mask_file, pts, color)   
  else:
    print('\nopen {}'.format(img_file))
    create_mask(mask_file, pts, color)