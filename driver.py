import classify
import parse
import graph
import argparse
import cv2
import sys


def main():
    formerequation = ''
    parse.primer()
    classify.primer()
    while 1:
        #cv2.imshow('Symbol Classification', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print('Exiting')
            break
        img = classify.load(sys.argv[1])# WILL'S CODE
        equation = classify.process(img, 0.1)
        if equation == formerequation:
            break

        func = parse.parse(equation)
        graph.graph(func,True)

if __name__ == '__main__':
    main()