import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from skimage.filters import sato
import matplotlib.pyplot as plt


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
    img[img<0.005] = 0 # remove background
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
def image_registration_without_label(img_original, img_filtered, img_reference):

    """create ORB detector with 5000 features"""
    orb_detector = cv2.ORB_create(5000)
    """find keypoints and descriptors"""
    keypoints1, descriptors1 = orb_detector.detectAndCompute(img_filtered, None)
    keypoints2, descriptors2 = orb_detector.detectAndCompute(img_reference, None)

    """match features between the two images"""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    """match the two sets of descriptors"""
    matches = matcher.match(descriptors1, descriptors2)
    number_of_matches = len(matches)

    """sort matches on the basis of their Hamming distance"""
    matches = sorted(matches, key=lambda x: x.distance)
    #matches = tuple(reversed(sorted(matches, key=lambda x: x.distance)))
    #matches = sort(key = lambda x: x.distance)
    """take the top 90% maches forward"""
    #if number_of_matches>10:
    #    matches = matches[:10]
    #    number_of_matches = len(matches)
    if number_of_matches>50:
        matches = matches[int(len(matches)*0.2):]
        #matches = matches[:int(len(matches)*0.8)]
        number_of_matches = len(matches)
    #matches = matches[:int(len(matches)*0.9)]
    print("Number of matches: ", number_of_matches)
    print("One match: ", matches[0])

    img_matches = cv2.drawMatches(img_filtered, keypoints1, img_reference, keypoints2, matches, img_reference, flags=2)

    """find the homography matrix"""
    p1 = np.zeros((number_of_matches,2))
    p2 = np.zeros((number_of_matches,2))
    for i in range(number_of_matches):
        p1[i,:] = keypoints1[matches[i].queryIdx].pt
        p2[i,:] = keypoints2[matches[i].trainIdx].pt
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    """transform the img1 wrt the reference im2"""
    #img_original = Image.open(path_original + img_filename.split('_')[1])
    #img_original = np.asarray(img_original)
    height, width = img_reference.shape
    img_transformed = cv2.warpPerspective(img_original, homography, (width, height))
    return img_transformed, img_matches


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
            plt.savefig(storage_folder + folder + "/reference_" + img_reference_name + ".jpg")
            img_reference_filtered = vessel_filter(img_reference)

            for idx in range(1,len(filenames)):
                img_name = filenames[idx]
                img = Image.open(path + img_name)
                img = np.asarray(img)
                img_filtered = vessel_filter(img)
                plt.figure()
                plt.imshow(img_filtered)
                plt.savefig(storage_folder + folder + "/filtered_" + img_name + ".jpg")

                img_aligned, img_matches = image_registration_without_label(img, img_filtered, img_reference_filtered)
                plt.figure()
                plt.imshow(img_aligned)
                plt.savefig(storage_folder + folder + "/aligned_" + img_name + ".jpg")

                plt.figure()
                plt.imshow(img_matches)
                plt.savefig(storage_folder + folder + "/matches_" + img_name + ".jpg")



        



