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
def image_registration_without_label(img_original, img_filtered, img_reference):
    radius = int(img_filtered.shape[0]/2) - 100

    """rotation and scaling"""
    img_filtered_polar = warp_polar(img_filtered, radius=radius)
    img_reference_polar = warp_polar(img_reference, radius=radius)
    shifts, error, diffphase = phase_cross_correlation(img_reference_polar, img_filtered_polar)
    shift_angle, shift_scale = shifts[:2]
    print("Shift scale: ", shift_scale)
    angle = (360/img_reference.shape[0]) * shift_angle
    print("Angle: ", angle)
    klog = radius / np.log(radius)
    scale = 1 / np.exp(shift_scale/klog)
    print("Scale: ", scale)

    print("Before transformation: ", img_original.shape)
    img_original_transformed = rotate(img_original, -shift_angle, resize=True)
    print("After transformation 1: ", img_original_transformed.shape)
    #img_original_transformed = rescale(img_original_transformed, 1/scale, preserve_range=True, channel_axis=2)
    print("After transformation 2: ", img_original_transformed.shape)
    img_filtered_transformed = rotate(img_filtered, -shift_angle, resize=True)
    #img_filtered_transformed = rescale(img_filtered_transformed, 1/scale, preserve_range=True)

    #if img_filtered.shape[0]>img_filtered_transformed.shape[0]:
    #    top = int((img_filtered.shape[0]-img_filtered_transformed.shape[0])/2)
    #    bottom = int(img_filtered.shape[0]-img_filtered_transformed.shape[0]-top)
    #    left = int((img_filtered.shape[1]-img_filtered_transformed.shape[1])/2)
    #    right = int(img_filtered.shape[1]-img_filtered_transformed.shape[1]-left)
    #    img_original_transformed = cv2.copyMakeBorder(img_original_transformed, top, bottom, left, right, cv2.BORDER_CONSTANT)
    #    img_filtered_transformed = cv2.copyMakeBorder(img_filtered_transformed, top, bottom, left, right, cv2.BORDER_CONSTANT)
    #    print("Type: ", type(img_filtered_transformed[0,0]))
    if img_filtered.shape[0]<img_filtered_transformed.shape[0]:
        marginx = int((img_filtered_transformed.shape[0] - img_filtered.shape[0])/2)
        marginy = int((img_filtered_transformed.shape[1] - img_filtered.shape[1])/2)
        img_original_transformed = img_original_transformed[marginx:marginx+img_filtered.shape[0],marginy:marginy+img_filtered.shape[1]]
        img_filtered_transformed = img_filtered_transformed[marginx:marginx+img_filtered.shape[0],marginy:marginy+img_filtered.shape[1]]
    img_filtered_transformed = cv2.normalize(src=img_filtered_transformed, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print("After transformation 3: ", img_original_transformed.shape)

    """schift"""
    shift, error, diffphase = phase_cross_correlation(img_reference, img_filtered_transformed)

    def shift_image(img, shift):
        dx, dy = shift
        print("Shift: ", dx, dy)
        dx = int(dx)
        dy = int(dy)
        Lx, Ly = img.shape
        img = np.roll(img, dx, axis=0)
        img = np.roll(img, dy, axis=1)
        if dx>0:
            img[:dx, :] = 0
        elif dx<0:
            img[Lx-dx:, :] = 0
        if dy>0:
            img[:, :dy] = 0
        elif dx<0:
            img[:, Ly-dy:] = 0
        return img

    img_original_transformed_0 = shift_image(img_original_transformed[:,:,0], shift)
    img_original_transformed_1 = shift_image(img_original_transformed[:,:,1], shift)
    img_original_transformed_2 = shift_image(img_original_transformed[:,:,2], shift)
    img_original_transformed = np.dstack([img_original_transformed_0, img_original_transformed_1, img_original_transformed_2])

    #print("transformed: ", img_transformed.shape)
    #print("reference: ", img_reference_original.shape)

    img_filtered_transformed = shift_image(img_filtered_transformed, shift)

    
    return img_original_transformed, img_filtered_transformed, angle, scale, shift


if __name__=='__main__':
    path = '/data/ml4s2021/data/2021_11_12/OphtalmoLaus/'

    storage_folder = 'output/'
    filenames_set = load_filenames_set(path)
    for folder in filenames_set.index:
        filenames = filenames_set[folder]
        if len(filenames)>1:
            os.makedirs(storage_folder + folder)
            img_reference_name = filenames[0]
            img_reference_original = Image.open(path + img_reference_name)
            img_reference_original = np.asarray(img_reference_original)
            plt.figure()
            plt.imshow(img_reference_original)
            plt.savefig(storage_folder + folder + "/reference_" + img_reference_name)
            img_reference_filtered = vessel_filter(img_reference_original)
            for idx in range(1,len(filenames)):
                img_name = filenames[idx]
                img_original = Image.open(path + img_name)
                img_original = np.asarray(img_original)
                print("DEBUG shape: ", img_original.shape, img_original[0].shape)
                img_filtered = vessel_filter(img_original)
                plt.figure()
                plt.imshow(cv2.addWeighted(img_filtered, 0.5, img_reference_filtered, 0.5, 0))
                plt.title("Reference + original (unaligned)")
                plt.savefig(storage_folder + folder + "/filtered_" + img_name)

                img_aligned, img_filtered_aligned, angle, scale, shift = image_registration_without_label(img_original, img_filtered, img_reference_filtered)
                plt.figure()
                plt.imshow(cv2.addWeighted(img_filtered_aligned, 0.5, img_reference_filtered, 0.5, 0))
                plt.title("Reference + aligned")
                plt.savefig(storage_folder + folder + "/aligned_" + img_name)

                plt.figure()
                plt.imshow(img_aligned)
                plt.title("Angle: " + str(angle) + ", scale: " + str(scale) + ", shift: " + str(shift))
                plt.savefig(storage_folder + folder + "/" + img_name)

        



