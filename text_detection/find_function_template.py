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
	if template_box is None:
		return None
	pics = []
	for t in template_box:
		print(t)
		pic = get_equation(image, t)
		if pic is None:
			continue
		pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
		pics.append(np.array(pic, dtype=np.uint8))
	return pics

def iterative_template_match(template, image):

	cv2.namedWindow('test', cv2.WINDOW_NORMAL)

	img = image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower_blue= np.array([78,158,124])
	upper_blue = np.array([138,255,255])

	mask = cv2.inRange(img,lower_blue,upper_blue)

	blues = np.where((img[...,0]>100)&(img[...,0]<135)&(img[...,2]>50)&(img[...,1]>25))

	mask = np.zeros(img.shape[:-1], dtype=np.uint8)
	mask[blues] = 255
	kernel = np.zeros((5,5),dtype = np.uint8)
	kernel[:,2] = 1
	mask = cv2.erode(mask,kernel,iterations=2)
	mask = cv2.dilate(mask,kernel,iterations=2)
	#cv2.imshow("test",mask)
	#cv2.waitKey()
	components = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
	stats = components[2]
	centroids = components[3]
	stats = np.append(stats, centroids, axis=1)
	combine = np.array(sorted(stats, key=lambda x: (x[4])))
	stats = list(combine[:, :-2])
	bluelist = []
	stats = stats[:-1]
	for stat in stats[::-1]:
		sleft, stop, swidth, sheight, sarea = stat
		if sarea > 100:
			bluelist.append((int(sleft),int(stop),int(sleft+swidth),int(stop+sheight)))
	#sleft, stop, swidth, sheight, sarea = stats[-1]
	#if sarea<100:
	#	return None
	print(bluelist)
	return  bluelist


def get_equation(img, template_box):

	sX, sY, eX, eY = template_box
	print (template_box)
	cropped = img[sY:eY, eX:, :]
	if len(cropped) == 0:
		return None
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

	if output[0]<2:
		return None
	imshow_components(output[1])
	box = chaincomponents(output)
	print(box)
	cv2.imshow('debug', cropped[int(box[1]):int(box[3]),int(box[0]):int(box[2])])
	#cv2.waitKey()
	return cropped[int(box[1]):int(box[3]),int(box[0]):int(box[2])]


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
		if sleft + 3*swidth > nleft and stop + 3*sheight > ntop:
			result[1] = min(ntop,result[1])
			result[2] = max(nleft+nwidth,result[2])
			result[3] = max(result[3],nheight+ntop)
		else:
			break
	return result


def imshow_components(labels, cuts=0):
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
	#cv2.waitKey()


if __name__ == '__main__':
	main()