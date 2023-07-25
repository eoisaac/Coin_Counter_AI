import cv2

video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))

    cv2.imshow("IMG", img)
    cv2.waitKey(1)

