
# Ce script nous permet de redéfinir la taille des images 
from PIL import Image
import numpy as np
import os



download_path = "C:/Users/yasin/OneDrive/Documents/projet_image_segmentation/resized_images_1/"
images_path = 'C:/Users/yasin/OneDrive/Documents/projet_image_segmentation/scrapper_dataset/dataset/trump/'

sqrWidth = 3024

listFileSplit = []
listFinal = []


for element in os.listdir(images_path):
	img_name_tab = element.split(".")
	img_name = img_name_tab[0]
	img_extension = img_name_tab[1]
	print(img_name)
	print(img_extension)
	im = Image.open(images_path+element)
	im_resize = im.resize((sqrWidth, sqrWidth))
	im_resize.save(download_path+img_name+'.'+img_extension)



