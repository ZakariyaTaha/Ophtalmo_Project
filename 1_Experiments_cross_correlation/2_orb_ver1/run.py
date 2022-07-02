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
def vessel_filter(path, img_filename):
    img = Image.open(path + img_filename)
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("DEBUG shape: ", img.shape)
    img = sato(img)
    img[img<0.005] = 0 # remove background

    def remove_circle(img):
        center = (int(img.shape[1]/2), int(img.shape[0]/2))
        radius = int(img.shape[0]/2) - 100
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)
        return cv2.bitwise_and(img, img, mask=mask)

    img = remove_circle(img)
    height, width = img.shape
    print("DEBUG1: ", height, width)
    return img


## Image registration, without label
def image_registration_without_label(path_filtered, img_filename, img_reference_filename, path_original):
    img_filtered_original = Image.open(path_filtered + img_filename)
    img_filtered_original = np.asarray(img_filtered_original)
    img_filtered = cv2.cvtColor(img_filtered_original, cv2.COLOR_BGR2GRAY)

    img_reference = Image.open(path_filtered + img_reference_filename)
    img_reference = np.asarray(img_reference)
    height, width, depth = img_reference.shape
    print("DEBUG: ", height, width, depth)
    img_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
    height, width = img_reference.shape
    print("DEBUG: ", height, width)

    """create ORB detector with 5000 features"""
    orb_detector = cv2.ORB_create(5000)
    """find keypoints and descriptors"""
    keypoints1, descriptors1 = orb_detector.detectAndCompute(img_filtered, None)
    keypoints2, descriptors2 = orb_detector.detectAndCompute(img_reference, None)
    """match features between the two images"""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Hamming distance = number of disagreement between two arrays
    """match the two sets of descriptors"""
    matches = matcher.match(descriptors1, descriptors2)
    #print(type(matches))
    """sort matches on the basis of their Hamming distance"""
    #matches.sort(key = lambda x: x.distance)
    """take the top 90% maches forward"""
    #matches = matches[:int(len(macthes)*0.9)]
    number_of_matches = len(matches)
    print("Number of matches: ", number_of_matches)
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
    img_transformed = cv2.warpPerspective(img_filtered_original, homography, (width, height))
    return img_transformed#cv2.addWeighted(img1_transformed, 0.5, img2_original, 0.5, 0)


if __name__=='__main__':
    path = '../../data/ml4s2021/data/2021_11_12/OphtalmoLaus/'

    storage_folder = 'filtered/'
    print("Filtering")
    if os.path.isdir(storage_folder)==False:
        filenames_set = load_filenames_set(path)
        for idx in filenames_set.index:
            os.makedirs(storage_folder + idx)
            for img in filenames_set[idx]:
                filtered_img = vessel_filter(path, img)
                plt.imshow(filtered_img)
                plt.savefig(storage_folder + idx + "/filtered_" + img)
    print("Filtering: Done")

    storage_folder = 'aligned/'
    print("Alignment")
    if os.path.isdir(storage_folder)==False:
        folders = [x for x in sorted(os.listdir('filtered/'))]
        for folder in folders:
            filenames = [x for x in sorted(os.listdir("filtered/" + folder)) if x.endswith('.jpg')]
            if len(filenames)>1:
                os.makedirs(storage_folder + folder)
                img_reference = Image.open(path + filenames[0].split('_')[1])
                img_reference = np.asarray(img_reference)
                plt.imshow(img_reference)
                plt.savefig(storage_folder + folder + "/reference_" + filenames[0])
                for idx in range(1,len(filenames)):
                    img_transformed = image_registration_without_label("filtered/" + folder + "/", filenames[idx], filenames[0], path)
                    plt.imshow(img_transformed)
                    plt.savefig(storage_folder + folder + "/aligned_" + filenames[idx])
                    
                
        



