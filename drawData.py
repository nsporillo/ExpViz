from preprocessing import *

import random

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
img = np.ones((540, 960, 3), np.uint8) * 255


# Mouse callback stroke function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.line(img, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=10)
            pt1_x, pt1_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, (pt1_x, pt1_y), (x, y), color=(0, 0, 0), thickness=10)


cv.namedWindow('Test Classification')
cv.setMouseCallback('Test Classification', line_drawing)

while 1:
    cv.imshow('Test Classification', img)
    key = cv.waitKey(1) & 0xFF
    rand = random.randint(5, 1000)
    if key == ord('s'):
        cv.imwrite('data/test' + str(rand) + '.jpg', img)
        # Reset the image in the drawing window
        img = np.ones((540, 960, 3), np.uint8) * 255
    elif key == ord('e'):
        print('Exiting')
        break

cv.destroyAllWindows()


