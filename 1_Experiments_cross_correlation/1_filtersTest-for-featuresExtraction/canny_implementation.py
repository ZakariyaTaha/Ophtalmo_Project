import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
from scipy import signal
import os
from PIL import Image
import math

path = '../../data/ml4s2021/data/2021_11_12/OphtalmoLaus/'

# Open the image
filename = [x for x in os.listdir(path) if x.endswith('.jpg')]

image = Image.open(path + filename[0]).convert('L')
#print(image.format)
#print(image.size)
#print(image.mode)
#plt.show()


image_arr = np.asarray(np.copy(image))
#print(type(image_arr))
#print(image_arr)

plt.imshow(image_arr)
plt.show()


M = 2


##########################################
def kernel(sigma, order, M):
	print('DEBUG: kernel')
	size = int(math.ceil(sigma*M)*2.0 + 1)
	kernel = np.zeros(size)

	center = math.ceil(M*sigma)

	if order==0:
		for k in range(size):
			kernel[k] = (1./(sigma*math.sqrt(2*math.pi))) * math.exp(-math.pow(k-center,2)/(2*math.pow(sigma,2)))
	elif order==1:
		for k in range(size):
			kernel[k] = -((k-center)/(math.pow(sigma,3)*math.sqrt(2*math.pi))) * math.exp(-math.pow(k-center,2)/(2*math.pow(sigma,2)))
	elif order==2:
		for k in range(size):
			kernel[k] = (math.pow(k-center,2)/(math.pow(sigma,5)*math.sqrt(2*math.pi))) * math.exp(-math.pow(k-center,2)/(2*math.pow(sigma,2)))

	return kernel



##########################################
def correlate(in_img, sigma, orderX, orderY, M):
	print('DEBUG: correlate')
	out = np.copy(in_img)
	kernelX = kernel(sigma, orderX, M)
	kernelY = kernel(sigma, orderY, M)
	size = len(kernelX)
	h = np.zeros((size, size))

	for i in range(len(kernelX)):
		for j in range(len(kernelY)):
			h[i][j] = kernelX[i] * kernelY[j]

	#for x in range(in_img.shape[0]):
	#	print('debug: ', x)
	#	for y in range(in_img.shape[1]):
	#		#b = np.zeros((size, size))
	#		tmp_size = int(math.floor(size/2))
	#		in_img_pad = np.pad(in_img, (tmp_size, tmp_size), mode='constant')
	#		b = in_img_pad[int(x+tmp_size-math.floor(size/2)):int(x+tmp_size+math.floor(size/2)+1), #int(y+tmp_size-math.floor(size/2)):int(y+tmp_size+math.floor(size/2)+1)]

	#		v = 0
	#		for i in range(size):
	#			for j in range(size):
	#				v = v + h[i][j]*b[i][j]

	#		out[x,y] = v
	out = scipy.signal.correlate(in_img, h, mode='same')

	return out


##########################################
def hessian(in_img, sigma):
	print('DEBUG: hessian')
	m = in_img
	ux = in_img
	uy = in_img

	inxx = correlate(in_img, sigma, 2, 0, M)
	inyy = correlate(in_img, sigma, 0, 2, M)
	inxy = correlate(in_img, sigma, 1, 1, M)

	for x in range(in_img.shape[0]):
		for y in range(in_img.shape[1]):
			h1 = inxx[x][y]
			h2 = inxy[x][y]
			h3 = h2
			h4 = inyy[x][y]

			a = 1
			b = -h1-h4
			c = h1*h4 - h2*h3

			delta = b*b - 4*a*c
			#print(delta)
			if delta<0:
				delta=0

			lambda_min = (-b - math.sqrt(delta)) / (2*a)
			lambda_max = (-b + math.sqrt(delta)) / (2*a)
			vx = h2
			vy = lambda_min - h1
			#print('vx: ', vx, ', vy: ', vy)
			v_norme = math.sqrt(vx*vx + vy*vy)
			m[x][y] = math.sqrt(abs(lambda_min)) * math.sqrt(abs(lambda_min-lambda_max))
			#print(vx)
			#print(v_norme)
			if vx==0.0:
				ux[x,y] = 0
			else:
				ux[x][y] = vx/v_norme
			if vy==0.0:
				uy[x,y] = 0
			else:
				uy[x][y] = vy/v_norme
	return m, ux, uy




##########################################
def applyNonMaximumSuppression(in_img):
	print('DEBUG: nms')
	out = np.copy(in_img[0])
	for x in range(in_img[0].shape[0]):
		for y in range(in_img[0].shape[1]):
			m = in_img[0]
			ux = in_img[1]
			#print(ux)
			uy = in_img[2]
			
			a = x+ux[x,y]
			b = y+uy[x,y]
			if a>=m.shape[0] or b>=m.shape[1]:
				msup = 1
			else:
				msup = m[x+ux[x][y]][y+uy[x][y]]

			a = x-ux[x,y]
			b = y-uy[x,y]
			if a<0 or b<0:
				minf = 0
			else:
				minf = m[x-ux[x][y]][y-uy[x][y]]
			if m[x][y]>=msup and m[x][y]>=minf:
				out[x][y] = m[x][y]
			else:
				out[x][y] = 0
	return out


##########################################
def hysteresisThreshold(nms, tL, tH):
	print('DEBUG: hysteresis')
	M, N = nms.shape
	out = np.zeros((M,N), dtype=np.uint8)
	
	strong_i, strong_j = np.where(nms>=tH)
	zeros_i, zeros_j = np.where(nms<tL)
	
	weak_i, weak_j = np.where(np.logical_and((nms<=tH),(nms>=tL)))

	out[strong_i, strong_j] = 255
	out[zeros_i, zeros_j] = 0
	out[weak_i, weak_j] = 75

	M, N = out.shape
	for i in range(1, M-1):
		for j in range(1, N-1):
			if 255 in [out[i+1, j-1], out[i+1, j], out[i+1, j+1], out[i, j-1], out[i, j+1], out[i-1, j-1], out[i-1, j], out[i-1, j+1]]:
				out[i,j] = 255
			else:
				out[i,j] = 0

	return out


##########################################
def ridge(in_img, sigma, tL, tH):
	print('DEBUG: ridge')
	hess = hessian(in_img, sigma)
	#nms = applyNonMaximumSuppression(hess)
	#out = hysteresisThreshold(nms, tL, tH)

	return hess[0]


image_out = ridge(image_arr, 1.2, 5.5, 10.5)

plt.imshow(image_out)
plt.show()

