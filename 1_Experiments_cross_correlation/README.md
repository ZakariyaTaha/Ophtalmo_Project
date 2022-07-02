This folder explores 6 techniques. The 4th one (cross-correlation) gives interesting results. The other ones are more experimentations.


# FOLDER 1: filtersTest-for-featuresExtraction

We tried different filters to extract the vessels (canny, frangi, meijering, sato)
applied on the image in particular color scales (gray scale, channel 0,1 or 2 of the color image)

## Observation

Best are meijering on channel 1, sato on gray scale

## How to run

script.py		to run
helpers.py		"implement" the vessel's filter
canny_implementation.py	attempt to implement canny filter by hand


# FOLDER 2: ORB version 1

Image registration with orb feature extractor

1. Image preprocessing
- filter the image (extract the vessels) -> sato on gray scale image
- remove the background -> set pixel value to 0 when its lower than 0.005
- remove the circle -> cv2.circle and cv2.bitwise_and
2. Image registration
- orb detector with 5000 features
- compute the homography matrix based on all the matches

## How to run

Output: 
filtered	preprocessing result
aligned		reference image (not moving, ground truth)
		aligned (filtered aligned images)

## Problem

When saving an image in png, its size changes (number of pixels) -> use jpg


# FOLDER 3: ORB version 2

Orb feature extractor, but
- no intermediate "filtered" file containing the filtered images
- transformation computed based on the filtered images as before, but applied on the color image

## How to run

Output:
reference	not moving image
filtered	filtered moving images
aligned		registered moving images

## Problem

The output images are too distorted


# FOLDER 4: Cross correlation

Image registration using cross-correlation
(rigid transformation: only compute the shift between two images)

## How to run

Output:
reference	original not moving image
"no prefix"	original moving images
filtered	filtered moving image
aligned		aligned moving images

## Problem

Trying to add rotation and scaling, but not working


# FOLDER 5: Sift feature extractor

Without preprocessing the images (no vessel extraction beforehand)

## How to run

Output:
matches		matches of the keypoints between original and moving image

## Problem

Deformation of the images.


# FOLDER 5: Orb version 3

Orb feature extractor with different parameters
(number of features and number of matches used to compute the homography matrix)

## How to run

Output:
matches		sometimes

## Problem

Deformation of the images












