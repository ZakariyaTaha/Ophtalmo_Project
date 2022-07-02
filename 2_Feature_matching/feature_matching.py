import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import skimage
from skimage import data, color, transform
from skimage.color import rgb2gray
from skimage.filters import meijering, sato, frangi, hessian
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches,
                             hessian_matrix, hessian_matrix_eigvals,
                             hog, SIFT)
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from skimage.measure import ransac
from skimage.io import imread, imshow, imsave

import cv2
import helpers

DATA_DIR = "/data/ml4s2021/data/2021_11_12/OphtalmoLaus/"
OUTPUT_DIR = "./full_output/"

images_df = helpers.load_df(DATA_DIR)

grouped_df = images_df.groupby(['patient_id', 'eye', 'centrage'], as_index=False)

target_df = grouped_df.first()
target_df[target_df['patient_id'] == '1']

df = pd.merge(
    target_df,
    images_df,
    how='inner',
    on=['patient_id', 'eye', 'centrage'],
    suffixes=('_target', '_source')
)
df = df[df['num_target'] != df['num_source']]

df['target-source-nmi'] = np.nan
df['target-moved-nmi'] = np.nan

last_loaded_path_target = ''

for idx, row in df.iterrows():
	print(idx)
	paths = row[['path_target', 'path_source']]
	if last_loaded_path_target == paths[0]:
		source = helpers.imp(DATA_DIR+paths[1], resize=False, vessels=True, exposure=True, cv=True)
	else:
		last_loaded_path_target == paths[0]
		target, source = map(
			lambda x: helpers.imp(DATA_DIR+x, resize=False, vessels=True, exposure=True, cv=True),
			paths
		)

	sift = cv2.SIFT_create(
		sigma=5,
		nfeatures=500,
		edgeThreshold=50,
	)

	target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
	source_keypoints, source_descriptors = sift.detectAndCompute(source, None)

	target_keypoints_arr = cv2.KeyPoint.convert(target_keypoints)
	target_keypoints_arr[:, 0], target_keypoints_arr[:, 1] = target_keypoints_arr[:, 1], target_keypoints_arr[:, 0].copy()
	source_keypoints_arr = cv2.KeyPoint.convert(source_keypoints)
	source_keypoints_arr[:, 0], source_keypoints_arr[:, 1] = source_keypoints_arr[:, 1], source_keypoints_arr[:, 0].copy()

	matches = match_descriptors(
		target_descriptors,
		source_descriptors,
		cross_check=True,
		metric='correlation'
	)

	src = source_keypoints_arr[matches[:, 1]][:, ::-1]
	tgt = target_keypoints_arr[matches[:, 0]][:, ::-1]

	model_robust, inliers = ransac((src, tgt), skimage.transform.SimilarityTransform,
		                          min_samples=10, residual_threshold=70,
		                          max_trials=500)

	moved = warp(
		source,
		model_robust.inverse,
		preserve_range=True,
		output_shape=target.shape,
		cval=-1,
	)
	moved = np.ma.array(moved, mask=moved==-1)

	target_source_nmi = skimage.metrics.normalized_mutual_information(target, source)
	target_moved_nmi = skimage.metrics.normalized_mutual_information(target, moved)

	df.loc[idx, 'target-source-nmi'] = target_source_nmi
	df.loc[idx, 'target-moved-nmi'] = target_moved_nmi

	print("Image mutual information")
	print(f"Target - Source : {target_source_nmi:.3f}")
	print(f"Target - Moved  : {target_moved_nmi:.3f}")

	target_img, source_img = imread(DATA_DIR+paths[0]), imread(DATA_DIR+paths[1])

	moved_img = warp(source_img, model_robust.inverse, preserve_range=True, output_shape=target.shape, cval=-1)
	moved_img = np.ma.array(moved_img, mask=moved_img==-1)

	folder = OUTPUT_DIR+f"{row['patient_id']}-{row['eye']}-{row['centrage']}"
	if not os.path.exists(folder):
		os.makedirs(folder, exist_ok=False)
	imsave(folder+f"/{row['path_target']}", target_img)
	imsave(folder+f"/{row['path_source']}", moved_img)

	print("MEAN Mutual information")
	print(f"Target - Source : {df['target-source-nmi'].mean():.3f}")
	print(f"Target - Moved  : {df['target-moved-nmi'].mean():.3f}")



