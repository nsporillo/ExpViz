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
	args = vars(ap.parse_args())

	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
	cv2.namedWindow('debug', cv2.WINDOW_NORMAL)


	template = cv2.imread(args["template"])

		# loop over the images to find the template in
	for imagePath in glob.glob(args["images"] + "/*.jpg"):
		image = cv2.imread(imagePath)
		template_box = iterative_template_match(template, image)
		get_equation(image, template_box)


def getSubImage(image,template):
	template_box = iterative_template_match(template, image)
	get_equation(image, template_box)

def iterative_template_match(template, image):

	template = cv2.pyrDown(template)
	template = cv2.pyrDown(template)
	template = cv2.pyrDown(template)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	template = cv2.Canny(template, 50, 200)
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

	return (startX, startY, endX, endY)


def get_equation(img, template_box):

	sX, sY, eX, eY = template_box
	cropped = img[sY:eY, eX:, :]
	cv2.imshow("Image", cropped)

	gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 200)
	ret, thresh = cv2.threshold(edges,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	kernel = np.ones((5,5),np.uint8)
	filled = cv2.dilate(thresh, kernel)
	# You need to choose 4 or 8 for connectivity type
	connectivity = 4  
	# Perform the operation
	output = cv2.connectedComponentsWithStats(filled, connectivity, cv2.CV_32S)
	imshow_components(output[1])
	box = chaincomponents(output)
	print(box)
	cv2.imshow('debug', cropped[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
	cv2.waitKey()


def chaincomponents(components):
	stats = components[2]
	centroids = components[3]
	stats = np.append(stats, centroids, axis=1)
	combine = np.array(sorted(stats, key=lambda x: (x[0], x[1])))
	centroids = list(combine[:, -2:])
	stats = list(combine[:, :-2])
	sleft, stop, swidth, sheight, sarea = stats[1][0], stats[1][1], stats[1][2], stats[1][3], stats[1][4]
	result = [sleft, stop, sleft+swidth, stop+sheight]
	for x in range(1,len(centroids)-1):
		sleft, stop, swidth, sheight, sarea = stats[x][0], stats[x][1], stats[x][2], stats[x][3], stats[x][4]
		if sarea < 50:
			continue
		n = centroids[x+1]
		c = centroids[x]
		nleft, ntop, nwidth, nheight, narea = stats[x+1][0], stats[x+1][1], stats[x+1][2], stats[x+1][3], stats[x+1][4]
		if sleft + 3*swidth > nleft:
			result[1] = min(ntop,result[1])
			result[2] = max(nleft+nwidth,result[2])
			result[3] = max(result[3],nheight+ntop)
		else:
			break
	return result

def imshow_components(labels):
	"""
	Display components throughout hue range.
	"""
	# Map component labels to hue val
	label_hue = np.uint8(179*labels/np.max(labels))
	blank_ch = 255*np.ones_like(label_hue)
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

	# cvt to BGR for display
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

	# set bg label to black
	labeled_img[label_hue==0] = 0

	cv2.imshow('debug', labeled_img)
	cv2.waitKey()


if __name__ == '__main__':
	main()