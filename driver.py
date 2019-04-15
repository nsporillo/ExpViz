import classify
import parse
import graph
import argparse
import cv2
import sys
import text_detection.find_function_template as td

def main():
    #get_new_template('text_detection/simple.jpg')
    formerequation = ''
    imgs = cv2.imread('data/test715.jpg')
    parse.primer()
    classify.primer()
    cv2.namedWindow('Exit Window',cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    frame = cap.read()[1]
    temp = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    while 1:
        cv2.imshow('Exit Window', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            print('Exiting')
            break
        frame = cap.read()[1]
        img = td.getSubImage(frame, temp)# WILL'S CODE
        if img is None:
            continue
        try:
            equation = classify.process(img, 0.1)
            if equation == formerequation:
                continue
            formerequation = equation
            func = parse.parse(equation)
            graph.graph(func,True)
            cv2.waitKey()
        except(Exception):
            print("Equation was not parsable")
            print("Equation " + equation)


def get_new_template(dest):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("New Template (press 'q' to ignore)", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        cv2.imshow("New Template (press 'q' to ignore)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(dest, frame)
            return

    cap.release()
    cv2.destroyWindow("New Template (press 'q' to ignore)")

if __name__ == '__main__':
    main()