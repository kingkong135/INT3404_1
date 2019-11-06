import cv2
import numpy as np
import os.path

# reference: https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/?fbclid=IwAR1Mjn-I_YHrQmeVqrQwom5IFWs95C7UUm1UATcNtdTQ2YCVdezojYHGA8c
def face_segment(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    mask = np.zeros(image.shape, np.uint8)
    for (x, y, w, h) in faces:
        mask[y :y + h + 30, x :x + w + 30] = image[y :y + h + 30, x :x + w + 30]
    converted = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    skinMask = cv2.erode(skinMask, kernel, iterations=3)
    skinMask = cv2.dilate(skinMask, kernel, iterations=5)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask=skinMask)

    cv2.imwrite('./output/out' + path, skin)


for path in os.listdir('./'):
    ext = os.path.splitext(path)[1]
    if ext.lower() == '.jpg':
        face_segment('' + path)
