import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

# construct the argument parse and parse the arguments
def argument():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help="path to the input image")
	ap.add_argument("-v", "--video", help="path to the input video")
	args = vars(ap.parse_args())
	return args

# canny edge detection => greyscale, blurred, edged
def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 50, 150)
	return edged

# create polygon ROI area for in front of lane
def region_of_interest(image):
	height = image.shape[0]
	width = image.shape[1]
	top_border = image.shape[0] / 2

	# left bot, left top, right top, right bot
	polygon = np.array([
		[(width*.1, height), (width / 2 - 25, top_border), (width / 2 + 25, top_border), (width*.9,height)]
		], dtype=np.int32)

	# Create mask. Fill in the polygon ROI in our mask. Perform bitwise_and on image+mask
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygon, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def display_lines(image,lines):
	# Create empty mask with same dimension as image input
	line_image = np.zeros_like(image)
	# For each line, draw line onto the mask
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

	# Combine image and mask
	combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

	return combined_image

# return average of all lines that belong to left lane line and all right lane line
def average_slope_intercept(image, lines):
	left_fit    = []
	right_fit   = []
	if lines is None:
		return None
	for line in lines:
		for x1, y1, x2, y2 in line:
			slope = (y2 - y1) / (x2 - x1)
			intercept = y1 - slope * x1
			if slope < -0.5 and slope != 0: # y is reversed in image
				left_fit.append((slope, intercept))
			elif slope > 0.5 and slope != 0:
				right_fit.append((slope, intercept))

	# Create an average line from the road lines we've encountered
	averaged_lines = None
	if len(left_fit) and len(right_fit):
		# average line for left
		left_fit_average  = np.average(left_fit, axis=0)
		# average line for right
		right_fit_average = np.average(right_fit, axis=0)
		# find intersection point of both lines if drawn all the way through
		intersection_point = find_intersection(left_fit_average[0], right_fit_average[0], left_fit_average[1], right_fit_average[1])
		# get points
		left_line  = make_points(image, left_fit_average, intersection_point)
		right_line = make_points(image, right_fit_average, intersection_point)
		averaged_lines = [left_line, right_line]

	return averaged_lines

# Find intersection point between the two lane line if drawn all the way through
def find_intersection(slope1, slope2, intercept1, intercept2):
	x = (intercept1-intercept2) /(slope2-slope1)
	y = slope1 * x + intercept1
	return [int(x),int(y)]

# Determine coordinates on how long to draw lane lines
def make_points(image, line, intersection):
	slope, intercept = line
	y1 = int(image.shape[0])# bottom of the image
	y2 = intersection[1] + 10
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return [[x1, y1, x2, y2]]

# Creates an average line from existing keyframe lines
def average_keyframes(keyframes, index):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for keyframe in keyframes:
        frame = keyframe[index]
        frame = frame[0]
        x1 += frame[0]
        y1 += frame[1]
        x2 += frame[2]
        y2 += frame[3]
    n = len(keyframes)
    return [[int(x1 / n), int(y1 / n), int(x2 / n), int(y2 / n)]]