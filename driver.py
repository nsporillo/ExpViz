import classify
import parse
import graph
import argparse
import cv2
import sys
from findFunction import getSubImage

def main():
    get_new_template('text_detection/simple.jpg')
    formerequation = ''
    imgs = cv2.imread('data/test715.jpg')
    parse.primer()
    classify.primer()
    cv2.namedWindow('Exit Window',cv2.WINDOW_NORMAL)

    testing_search_img = cv2.imread('text_detection/search_images/search.jpg')
    template = cv2.imread('text_detection/simple.jpg')
    while 1:
        cv2.imshow('Exit Window', imgs)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print('Exiting')
            break
        equation_path = getSubImage(testing_search_img, template)
        equation_img = cv2.imread(equation_path)
        cv2.imshow('Exit Window', equation_img)
        cv2.waitKey(0)
        img = classify.load(equation_path)
        equation = classify.process(img, 0.1, True)
        if equation == formerequation:
            continue
        formerequation = equation
        func = parse.parse(equation)
        graph.graph(func,True)


def get_new_template(dest):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("New Template", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        cv2.imshow("New Template", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(dest, frame)
            return

    cap.release()
    cv2.destroyWindow("New Template")


if __name__ == '__main__':
    main()