import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import data, color, transform
from skimage.color import rgb2gray
from skimage.filters import meijering, sato, frangi, hessian
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches,
                             hessian_matrix, hessian_matrix_eigvals,
                             hog)
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from skimage.measure import ransac
from skimage.io import imread, imshow, imsave

import cv2


def preprocess_img(raw_img):
	img = rgb2gray(raw_img)
	denoised_img = skimage.restoration.denoise_bilateral(img)
	equalized_img = skimage.exposure.equalize_adapthist(
		denoised_img,
		kernel_size=None,
		clip_limit=0.05
	)
	return equalized_img


def load_df(path):
	pattern = "^(\d+)-(\d+)-([LR])-(cfi|multicolor|other)-(OHN|macula|LQ)-(OphtalmoLaus)-(\d+)\.jpg"
	filenames = [x for x in sorted(os.listdir(path)) if x.endswith('.jpg')]
	def extract(x):
		a = re.search(pattern, x)
		if a is None or not a:
		    return None
		return a.groups()
	extracted = list(map(extract, filenames))
	images_df = pd.DataFrame(extracted, columns=['patient_id', 'date', 'eye', 'image_type', 'centrage', 'dataset', 'num'])
	images_df['path'] = filenames
	images_df.dropna(how='any', inplace=True)
	images_df.sort_values(['patient_id', 'eye', 'centrage', 'num'], inplace=True)
	images_df.drop_duplicates(['patient_id', 'eye', 'image_type', 'centrage', 'dataset', 'num'], inplace=True)
	images_df.drop(['date', 'image_type', 'dataset'], axis=1, inplace=True)
	return images_df

def resize_mask(mask, scale, padding, crop=None):
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=1)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y+h, x:x+w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def squarize(img, size):
    h, w = img.shape   
    top_pad = (size-h) // 2
    bottom_pad = size - h - top_pad
    left_pad = (size - w) // 2
    right_pad = size - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    img = np.pad(img, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h+top_pad, w+left_pad)
    return img

def imp(path, resize=True, vessels=False, exposure=False, cv=False):
	if cv:
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	else:
		img = imread(path)
		img = rgb2gray(img)
	#img = preprocess_img(img)
	#img = meijering(img)
	if vessels:
		img = sato(img)
		img[img<0.0025] = 0 # remove background
		#img[img>=0.01] = 255
		img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		
		def remove_circle(img):
			center = (int(img.shape[1]/2), int(img.shape[0]/2))
			radius = int(img.shape[0]/2) - 100
			mask = np.zeros(img.shape[:2], dtype="uint8")
			cv2.circle(mask, center, radius, 255, -1)
			return cv2.bitwise_and(img, img, mask=mask)
		
		img = remove_circle(img)
		img = skimage.exposure.rescale_intensity(img)
	if resize:
		img = cv2.resize(img, dsize=(512, 512))
	if exposure and not vessels:
		img = skimage.exposure.rescale_intensity(img)
	return(img)

def vxm_data_generator(sources, targets, batch_size=8):
    
    vol_shape = targets.shape[1:]
    ndims = len(vol_shape)
    
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        idx1 = np.random.randint(0, targets.shape[0], size=batch_size)
        sources_images = sources[idx1, ..., np.newaxis]
        targets_images = targets[idx1, ..., np.newaxis]
        inputs = [sources_images, targets_images]
        
        outputs = [targets_images, zero_phi]
        
        yield (inputs, outputs)

def plot_history(hist):
	plt.figure()
	plt.plot(hist.epoch, hist.history['loss'], '.-')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.show()
