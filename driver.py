import classify
import parse
import graph
import argparse
import cv2
import sys

def main():
    formerequation = ''
    imgs = cv2.imread('data/test715.jpg')
    parse.primer()
    classify.primer()
    cv2.namedWindow('Exit Window',cv2.WINDOW_NORMAL)
    while 1:
        cv2.imshow('Exit Window', imgs)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print('Exiting')
            break
        img = classify.load(sys.argv[1])# WILL'S CODE
        equation = classify.process(img, 0.1)
        if equation == formerequation:
            continue
        formerequation = equation
        func = parse.parse(equation)
        graph.graph(func,True)

if __name__ == '__main__':
    main()