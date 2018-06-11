import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
	"""
	Takes an image, gradient orientation, and threshold min/max values
	"""
	# Convert to grayscale
	if img.shape.__len__() == 3:
		# Convert to grayscale转换为灰度
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		gray = img
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	if img.shape.__len__() == 3:
		# Convert to grayscale转换为灰度
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		gray = img
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

	# Return the binary image
	return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	Return the direction of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Convert to grayscale
	if img.shape.__len__() == 3:
		# Convert to grayscale转换为灰度
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else:
		gray = img
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Take the absolute value of the gradient direction,
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output


def hls_thresh(img, thresh=(70, 255)):
	"""
	Convert RGB to HLS and threshold to binary image using S channel
	"""
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	return binary_output

# 在HSV空间下进行白色和黄色的筛选
def hls_thresh2(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H, V, S = cv2.split(converted)
    # cv2.imshow('H', H)
    # cv2.imshow('S', S)
    # cv2.imshow('V', V)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([50, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower1 = np.uint8([10,  0, 90])
    upper1 = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(converted, lower1, upper1)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    binary_output = np.zeros_like(white_mask)
    binary_output[mask>0] = 1
    # cv2.imshow('mask', binary_output)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # binary_output_closed = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel)
    # binary_output_opened = cv2.morphologyEx(binary_output_closed, cv2.MORPH_OPEN, kernel)
    # moropen = cv2.dilate(binary_output, kernel)
    # cv2.imshow('moropen', moropen)
    # cv2.imshow('binary_output', binary_output)

    # fig = plt.gcf()
    # fig.set_size_inches(16.5, 8.5)
    # plt.subplot(2, 3, 1)
    # plt.imshow(H, cmap='gray', vmin=0, vmax=255)
    # plt.title('H')
    # plt.subplot(2, 3, 2)
    # plt.imshow(S, cmap='gray', vmin=0, vmax=255)
    # plt.title('S')
    # plt.subplot(2, 3, 3)
    # plt.imshow(V, cmap='gray', vmin=0, vmax=255)
    # plt.title('V')
    # plt.subplot(2, 3, 4)
    # plt.imshow(moropen, cmap='gray', vmin=0, vmax=1)
    # plt.title('moropen')
    # plt.subplot(2, 3, 5)
    # plt.imshow(white_mask, cmap='gray', vmin=0, vmax=1)
    # plt.title('white_mask')
    # plt.subplot(2, 3, 6)
    # plt.imshow(yellow_mask, cmap='gray', vmin=0, vmax=1)
    # plt.title('yellow_mask')
    # plt.tight_layout()
    # plt.show()
    return binary_output


def combined_thresh(img):
	blur = cv2.bilateralFilter(img, 9, 75, 75)
	img = cv2.GaussianBlur(blur, (5, 5), 0)
	converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	imgH, imgV, imgS = cv2.split(converted)
	abs_bin_V = abs_sobel_thresh(imgV, orient='x', thresh_min=50, thresh_max=255)
	abs_bin_S = abs_sobel_thresh(imgS, orient='x', thresh_min=20, thresh_max=255)
	mag_bin = mag_thresh(imgS, sobel_kernel=3, mag_thresh=(25, 255))
	dir_bin = dir_threshold(imgS, sobel_kernel=15, thresh=(0.7, 1.3))
	# hls_bin = hls_thresh(img, thresh=(90, 255))
	hls_bin = hls_thresh2(img)

	combined = np.zeros_like(abs_bin_V)
	combined_abs_bin = np.zeros_like(abs_bin_V)
	combined_abs_bin[(abs_bin_S == 1) | abs_bin_V == 1 ] = 1
	combined[(((abs_bin_S == 1) & mag_bin == 1) | abs_bin_V == 1) | hls_bin== 1] = 1

	# fig = plt.gcf()
	# fig.set_size_inches(16.5, 8.5)
	# plt.subplot(2, 3, 1)
	# plt.imshow(abs_bin_V, cmap='gray', vmin=0, vmax=1)
	# plt.title('abs_bin_V')
	# plt.subplot(2, 3, 2)
	# plt.imshow(abs_bin_S, cmap='gray', vmin=0, vmax=1)
	# plt.title('abs_bin_S')
	# plt.subplot(2, 3, 3)
	# plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
	# plt.title('mag_bin')
	# plt.subplot(2, 3, 4)
	# plt.imshow(combined1, cmap='gray', vmin=0, vmax=1)
	# plt.title('combined1')
	# plt.subplot(2, 3, 5)
	# plt.imshow(hsv_bin, cmap='gray', vmin=0, vmax=1)
	# plt.title('hsv_bin')
	# plt.subplot(2, 3, 6)
	# plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
	# plt.title('combined')
	# plt.tight_layout()
	# plt.show()
	return combined, combined_abs_bin, mag_bin, dir_bin, hls_bin  # DEBUG


if __name__ == '__main__':
	#img_file = 'test_images/straight_lines1.jpg'
	img_file = 'test_images/test1.jpg'

	with open('calibrate_camera.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	img = mpimg.imread(img_file)
	img = cv2.undistort(img, mtx, dist, None, mtx)

	combined, abs_bin_V, abs_bin_S, dir_bin, hls_bin = combined_thresh(img)

	plt.subplot(2, 3, 1)
	plt.imshow(abs_bin_V, cmap='gray', vmin=0, vmax=1)
	plt.title('abs_bin_V')
	plt.subplot(2, 3, 2)
	plt.imshow(abs_bin_S, cmap='gray', vmin=0, vmax=1)
	plt.title('abs_bin_S')
	plt.subplot(2, 3, 3)
	plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
	plt.title('dir_bin')
	plt.subplot(2, 3, 4)
	plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
	plt.title('hls_bin')
	plt.subplot(2, 3, 5)
	plt.imshow(img)
	plt.title('img')
	plt.subplot(2, 3, 6)
	plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
	plt.title('combined')

	plt.tight_layout()
	plt.show()
