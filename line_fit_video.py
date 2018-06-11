import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset, viz2
from moviepy.editor import VideoFileClip


# Global variables (just to make the moviepy video annotation work)
with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature
frameCount = 0
retLast = {}


# MoviePy video annotation will call this function
def annotate_image(img_in):
	"""
	Annotate the input image with lane line markings
	Returns annotated image
	"""
	global mtx, dist, left_line, right_line, detected, frameCount, retLast
	global left_curve, right_curve, left_lane_inds, right_lane_inds

	frameCount += 1
	src = np.float32(
		[[200, 720],
		 [1100, 720],
		 [520, 500],
		 [760, 500]])

	x = [src[0, 0], src[1, 0], src[3, 0], src[2, 0], src[0, 0]]
	y = [src[0, 1], src[1, 1], src[3, 1], src[2, 1], src[0, 1]]

	# Undistort, threshold, perspective transform
	undist = cv2.undistort(img_in, mtx, dist, None, mtx)
	img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
	binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)



	# Perform polynomial fit
	if not detected:
		# Slow line fit
		ret = line_fit(binary_warped)
		# if detect no lanes, use last result instead.
		if len(ret) == 0:
			ret = retLast
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		out_img = ret['out_img']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']
		histogram = ret['histo']

		# Get moving average of line fit coefficients
		left_fit = left_line.add_fit(left_fit)
		right_fit = right_line.add_fit(right_fit)

		# Calculate curvature
		left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

		detected = True  # slow line fit always detects the line

	else:  # implies detected == True
		# Fast line fit
		left_fit = left_line.get_fit()
		right_fit = right_line.get_fit()
		ret = tune_fit(binary_warped, left_fit, right_fit)
		left_fit = ret['left_fit']
		right_fit = ret['right_fit']
		nonzerox = ret['nonzerox']
		nonzeroy = ret['nonzeroy']
		left_lane_inds = ret['left_lane_inds']
		right_lane_inds = ret['right_lane_inds']

		# Only make updates if we detected lines in current frame
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']

			left_fit = left_line.add_fit(left_fit)
			right_fit = right_line.add_fit(right_fit)
			left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
		else:
			detected = False

	vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

	# Perform final visualization on top of original undistorted image
	result = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

	retLast = ret

	save_viz2 = './output_images/polyfit_test%d.jpg' % (frameCount)

	viz2(binary_warped, ret, save_viz2)

	save_warped = './output_images/warped_test%d.jpg' % (frameCount)
	plt.imshow(binary_warped, cmap='gray', vmin=0, vmax=1)
	if save_warped is None:
		plt.show()
	else:
		plt.savefig(save_warped)
	plt.gcf().clear()

	save_binary = './output_images/binary_test%d.jpg' % (frameCount)
	plt.imshow(img,	cmap='gray', vmin=0, vmax=1)
	if save_binary is None:
		plt.show()
	else:
		plt.savefig(save_binary)
	plt.gcf().clear()

	if frameCount > 0:
		fig = plt.gcf()
		fig.set_size_inches(16.5, 8.5)
		plt.subplot(2, 3, 1)
		plt.imshow(undist)
		# plt.plot(undist)
		plt.plot(x, y)
		plt.title('undist')
		plt.subplot(2, 3, 2)
		plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
		plt.title('hls_bin')
		plt.subplot(2, 3, 3)
		plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
		plt.title('abs_bin')
		plt.subplot(2, 3, 4)
		plt.imshow(img, cmap='gray', vmin=0, vmax=1)
		plt.title('img')
		plt.subplot(2, 3, 5)
		plt.imshow(out_img)
		plt.title('out_img')
		plt.subplot(2, 3, 6)
		plt.imshow(result, cmap='gray', vmin=0, vmax=1)
		plt.title('result')

		save_result = 'D:/code/github_code/CarND-Advanced-Lane-Lines-P4/output_images/result-test%d.jpg' % (frameCount)
		if save_result is None:
			plt.show()
		else:
			plt.savefig(save_result)
		plt.gcf().clear()

	return result


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	# Annotate the video
	# annotate_video('challenge_video.mp4', 'challenge_video_out.mp4')

	# Show example annotated image on screen for sanity check
	for i in range (1, 7):

		img_file = 'test_images/test%d.jpg' % (i)
		img = mpimg.imread(img_file)
		result = annotate_image(img)
		plt.imshow(result)
		save_file = 'D:/code/github_code/CarND-Advanced-Lane-Lines-P4/output_images/test%d.jpg' % (i)
		if save_file is None:
			plt.show()
		else:
			plt.savefig(save_file)
		plt.gcf().clear()
