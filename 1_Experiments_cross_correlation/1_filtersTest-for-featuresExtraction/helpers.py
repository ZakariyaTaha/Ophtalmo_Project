import cv2
import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


######################################################################
## Separate data into pandas series
## (with same id, same eye, same centerness)
######################################################################

def extractIdEyeCenterness(name):
	decomp = name.split('-')

	return decomp[0] + decomp[3] + decomp[5]


## Convert the list of filenames into pandas series of same id, same eye and same centerness
def toSet(filename_list):
	df = pd.DataFrame({'name': filename_list, 'decomp': [extractIdEyeCenterness(elem) for elem in filename_list]})

	return df.groupby('decomp')['name'].apply(list)


## Load the filenames in sets of same id, same eye and same centerness
def loadFilenames(path):
	filename = [x for x in sorted(os.listdir(path)) if x.endswith('.jpg')]	# filenames in alphabetical order
	filenameSet = toSet(filename)

	return filenameSet


######################################################################
## Extract the vessels of an image
######################################################################

from skimage.filters import sato
from skimage.filters import frangi
from skimage.filters import meijering


def vessel_filter(path, img_filename):
	img = Image.open(path + img_filename)
	img_arr = np.asarray(img)[:,:,2]
	#img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
	#img_arr = cv2.GaussianBlur(img_arr, (3,3), 0)
	plt.imshow(img_arr)
	plt.savefig("original_channel2.png")

	canny_img = cv2.Canny(img_arr, threshold1=20, threshold2=50)
	plt.imshow(canny_img)
	plt.savefig("canny_channel2.png")

	#sigma = 0.33
	#median = np.median(img_arr)
	#lower = int(max(0, (1.0-sigma)*median))
	#upper = int(min(255, (1.0+sigma)*median))
	#auto_canny_img = cv2.Canny(img_arr, lower, upper)
	#plt.imshow(auto_canny_img)
	#plt.savefig("auto_canny_img_gray+blurred.png")

	sato_img = sato(img_arr)
	plt.imshow(sato_img)
	plt.savefig("sato_channel2.png")

	frangi_img = frangi(img_arr)
	plt.imshow(frangi_img)
	plt.savefig("frangi_channel2.png")

	meijering_img = meijering(img_arr)
	plt.imshow(meijering_img)
	plt.savefig("meijering_channel2.png")


