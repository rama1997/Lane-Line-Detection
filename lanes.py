"""
Things to change to adjust line detection
- parameters of lines = cv2.HoughLinesP
- past keyframes limit when averaging
- region of interest
- slope of lines
- make_point
"""
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import helper

# Get any arguments
args = helper.argument()

# Load image
image = args["image"]
if image is not None:
	original = cv2.imread(args["image"])
	keyframes = []
	# Operate on a copy so we do not mutate original frame
	image = np.copy(original)
	# Canny edge detect
	edged = helper.canny(image)
	# Get ROI
	cropped = helper.region_of_interest(edged)
	# Find lines in ROI
	lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 85, np.array([]), minLineLength = 40, maxLineGap = 5)
	# find average line between all lines found
	averaged_lines = helper.average_slope_intercept(image, lines)
	# Display lines onto image
	line_image = helper.display_lines(image, averaged_lines)

	cv2.imshow("Result", line_image)
	cv2.waitKey(0)

# video logic
old_lines = None
video = args["video"]
keyframes = []
post_frames = []
size = (0,0)
if video is not None:
	cap = cv2.VideoCapture(args["video"])
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			# Canny edge detect
			edged = helper.canny(frame)
			# get ROI only
			cropped = helper.region_of_interest(edged)
			# find lines in ROI
			lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
			# find average line between all lines found
			averaged_lines = helper.average_slope_intercept(frame, lines)

			# Add our line to keyframes to keep a rolling average for smoothing
			if averaged_lines:
				keyframes.append([averaged_lines[0], averaged_lines[1]])

				# Average together no more than 5 past frames
				if len(keyframes) > 5:
					keyframes.pop(0)

			if len(keyframes) > 0:
				left_line = helper.average_keyframes(keyframes, 0)
				right_line = helper.average_keyframes(keyframes, 1)
				line_image = helper.display_lines(frame, [left_line, right_line])
			else:
				line_image = helper.display_lines(frame, averaged_lines)

			# Adding the new frame into array and storing size for output video creation later
			post_frames.append(line_image)
			height, width, layers = frame.shape
			size = (width,height)

			cv2.imshow("result", line_image)
			# Show every millisecond and end with keyboard key q
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	# Creating output video
	output_vid = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15, size)
	for i in range(len(post_frames)):
	    output_vid.write(post_frames[i])
	output_vid.release()

	cap.release()
	cv2.destroyAllWindows()
