import cv2


class CoinDetector:
    def __init__(self):
        self._video = cv2.VideoCapture(0)
        pass

    def capture_video(self):
        try:
            while True:
                on, img = self._video.read()

                if on:
                    img = cv2.resize(img, (640, 480))
                    cv2.imshow("IMG", img)
                    cv2.waitKey(1)

        except KeyboardInterrupt:
            exit(0)

        except Exception as e:
            print(e)
