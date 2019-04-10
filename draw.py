import pickle

from keras.models import model_from_yaml

from preprocessing import *

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
width, height = 224, 224
img = np.zeros((height, width, 3), np.uint8)


def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


# Mouse callback stroke function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)
            pt1_x, pt1_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=10)

"""
Test program for drawing in characters and observing the prediction and confidence
Press 'c' after drawing a character to predict the class
Press 'e' when you are done and want to exit.

"""
model = load_model('bin')
mapping = pickle.load(open('bin/mapping.p', 'rb'))

cv.namedWindow('Symbol Classification')
cv.setMouseCallback('Symbol Classification', line_drawing)

history = list()

while 1:
    if len(history) > 0:
        hist = np.hstack(history)
        cv.imshow('Symbol Classification', np.hstack((img, hist)))
    else:
        cv.imshow('Symbol Classification', img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('c'):
        # Convert to grayscale, downsample by a factor of 4 and normalize intensities before prediction
        pred = cv.cvtColor(cv.resize(img, (28, 28)), cv.COLOR_BGR2GRAY).reshape(1, 28, 28, 1).astype('float32') / 255

        # Predict and print
        result = model.predict(pred)
        index = int(np.argmax(result, axis=1)[0])
        if index < len(mapping):
            print("Pred= [" + mapping[index] + "] conf= " + str(round(max(result[0]) * 100, 2)))
            img = cv.putText(img, mapping[index], (90, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))
            img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=(255, 0, 0))
            img = cv.resize(img, (height, width))
            history.append(img.copy())
            if len(history) >= 6:
                history.pop(0)
        else:
            print('No prediction for mapping[' + str(index) + ']')
        # Reset the image in the drawing window
        img = np.zeros((height, width, 3), np.uint8)
    elif key == ord('e'):
        print('Exiting')
        break

cv.destroyAllWindows()


