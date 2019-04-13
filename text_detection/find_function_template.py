import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import imutils
import glob


def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--template", required=True, help="Path to template image")
	ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
	ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration", action="store_true")
	args = vars(ap.parse_args())

	cv2.namedWindow('Visualize', cv2.WINDOW_NORMAL)
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
	cv2.namedWindow('Template', cv2.WINDOW_NORMAL)

	template = cv2.imread(args["template"])
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	template = cv2.Canny(template, 50, 200)

	cv2.imshow("Template", template)

		# loop over the images to find the template in
	for imagePath in glob.glob(args["images"] + "/*.jpg"):
		image = cv2.imread(imagePath)
		iterative_template_match(template, image)


def iterative_template_match(template, image):

	(tH, tW) = template.shape[:2]

	# convert the image to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
 
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
 
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
		# if we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
 
	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
	# draw a bounding box around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)


def get_equation(img, template_box):
	pass


if __name__ == '__main__':
	main()