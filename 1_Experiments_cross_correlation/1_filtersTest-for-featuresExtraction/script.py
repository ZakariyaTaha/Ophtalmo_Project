import helpers
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#import helpers2


######################################################################
## Import the images in sets of same id, same eye and same centerness
######################################################################

path = '../../data/ml4s2021/data/2021_11_12/OphtalmoLaus/'
filenameSet = helpers.loadFilenames(path)
#print(filenameSet)


######################################################################
## Determine the best channel (where the vessels are the most visible)
######################################################################

#img = Image.open(path + filenameSet[0][0])#.convert('L')
#print(img.format)
#print(img.size)
#print(img.mode)

# Convert image as array
#img_arr = np.asarray(img)

# Display the image in each channel
#plt.subplot(131)
#plt.imshow(img_arr[:,:,0])
#plt.title('Dimension 0')

#plt.subplot(132)
#plt.imshow(img_arr[:,:,1])
#plt.title('Dimension 1')

#plt.subplot(133)
#plt.imshow(img_arr[:,:,2])
#plt.title('Dimension 2')

#plt.show()
#plt.savefig("test1.png")

# The vessels are the most visible in channel 2 (dimension 1)


######################################################################
## Extract the vessels
######################################################################

#img_original, img_edges, img_combine = helpers.cannyRidge_detector(path, filenameSet[0][0])
#helpers.auto_canny(path, filenameSet[0][0], sigma=0.1)
#print(img_original.shape)


#print(type(filenameSet[0][0]))
helpers.vessel_filter(path, filenameSet[0][0])
#print(result)

