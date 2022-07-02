import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from skimage.filters import sato
import matplotlib.pyplot as plt
from skimage.transform import warp_polar, rotate, rescale
from skimage.registration import phase_cross_correlation


## Load the filenames in sets of same id, same eye and same centerness
def load_filenames_set(path):
    def extract_IdEyeCenterness(name):
        decomp = name.split('-')
        return decomp[0] + '_' + decomp[2] + '_' + decomp[4]

    def to_set(filenames_list):
        """convert the list of filenames into pandas series of same id, same eye and same centerness"""
        df = pd.DataFrame({'name': filenames_list, 'decomp': [extract_IdEyeCenterness(elem) for elem in filenames_list]})
        return df.groupby('decomp')['name'].apply(list)

    filenames = [x for x in sorted(os.listdir(path)) if x.endswith('.jpg')] # filenames in alphabetical order
    filenames_set = to_set(filenames)
    return filenames_set


## Extract the vessels of an image
def vessel_filter(img: np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Debug type: ", type(img[0,0]))
    print("DEBUG shape: ", img.shape)
    img = sato(img)
    print("Debug type: ", type(img[0,0]))
    img[img>0.006] = 255
    img[img<0.006] = 0 # remove background
    print("Debug type: ", type(img[0,0]))
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def remove_circle(img):
        center = (int(img.shape[1]/2), int(img.shape[0]/2))
        radius = int(img.shape[0]/2) - 100
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)
        return cv2.bitwise_and(img, img, mask=mask)

    img = remove_circle(img)
    height, width = img.shape
    print("DEBUG1: ", height, width)
    print("Debug type: ", type(img[0,0]))
    return img


## Image registration, without label
def image_registration_without_label(img, img_reference):
    """convert images to gray scale"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_reference_gray = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)

    """determine the keypoints"""
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_reference_gray, None)

    print("Number of keypoints: ", len(keypoints_1), "in img, ", len(keypoints_2), "in reference")

    """match the features"""
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    number_of_matches = len(matches)
    print("Number of matches: ", number_of_matches)

    img_matches = cv2.drawMatches(img_gray, keypoints_1, img_reference_gray, keypoints_2, matches, img_reference_gray, flags=2)

    """find the homography matrix"""
    p1 = np.zeros((number_of_matches,2))
    p2 = np.zeros((number_of_matches,2))
    for i in range(number_of_matches):
        p1[i,:] = keypoints_1[matches[i].queryIdx].pt
        p2[i,:] = keypoints_2[matches[i].trainIdx].pt
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    """transform the img1 wrt the reference im2"""
    height, width = img_reference_gray.shape
    img_transformed = cv2.warpPerspective(img, homography, (width, height))  
    
    return img_matches, img_transformed


if __name__=='__main__':
    path = '../../data/ml4s2021/data/2021_11_12/OphtalmoLaus/'

    storage_folder = 'output/'
    filenames_set = load_filenames_set(path)

    for folder in filenames_set.index:
        filenames = filenames_set[folder]
        if len(filenames)>1:
            os.makedirs(storage_folder + folder)
            img_reference_name = filenames[0]
            img_reference = Image.open(path + img_reference_name)
            img_reference = np.asarray(img_reference)
            plt.figure()
            plt.imshow(img_reference)
            plt.savefig(storage_folder + folder + "/reference_" + img_reference_name)

            for idx in range(1,len(filenames)):
                img_name = filenames[idx]
                img = Image.open(path + img_name)
                img = np.asarray(img)
                plt.figure()
                plt.imshow(img)
                plt.savefig(storage_folder + folder + "/" + img_name)

                img_matches, img_aligned = image_registration_without_label(img, img_reference)
                plt.figure()
                plt.imshow(img_matches)
                plt.savefig(storage_folder + folder + "/matches_" + img_name)

                plt.figure()
                plt.imshow(img_aligned)
                plt.savefig(storage_folder + folder + "/aligned_" + img_name)

        



