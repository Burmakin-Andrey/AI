import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model

model = Sequential()
model = load_model("mnist.h5")

img = np.zeros((512,512), dtype="uint8")

cv2.namedWindow("WHIST")

draw = False


def draw_callback(event,x,y,flags,param):
    global draw
    if event == cv2.EVENT_MOUSEMOVE:
        if draw:
            cv2.circle(img,(x,y),5,200,-1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False


cv2.setMouseCallback("WHIST",draw_callback)

while True:
    cv2.imshow("WHIST", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('s'):
        image = cv2.resize(img.copy() / 255, (28, 28))
        image = image.reshape(1,28,28,1)
        predictions = model.predict(image)
        print(np.argmax(predictions, 1))
        print(predictions)
    if key == ord("c"):
        print("Clear")
        img[:] = 0

cv2.destroyAllWindows()