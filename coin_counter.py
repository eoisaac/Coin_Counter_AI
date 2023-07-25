import cv2
import numpy as np
from termcolor import colored


class CoinCounter:
    def __init__(self):
        self._min_coin_area = 2000
        self._video = cv2.VideoCapture(0)
        pass

    def _video_pre_processing(self, img):
        pre_img = cv2.GaussianBlur(img, (5, 5), 3)
        pre_img = cv2.Canny(pre_img, 90, 140)

        kernel = np.ones((4, 4), np.uint8)
        pre_img = cv2.dilate(pre_img, kernel, iterations=2)
        pre_img = cv2.erode(pre_img, kernel, iterations=1)
        return pre_img

    def _capture_video(self, width=640, height=480, pre_processing=True):
        img, pre_img = None, None

        has_video, img = self._video.read()
        if has_video:
            img = cv2.resize(img, (width, height))
            if pre_processing:
                pre_img = self._video_pre_processing(img)

        return img, pre_img

    def _show_video(self, img, pre_img):
        cv2.imshow("Source", img)
        cv2.imshow("Pre-processed", pre_img)
        cv2.waitKey(1)

    def count_coins(self):
        print(colored("Counting coins...", "green", attrs=["bold"]))
        try:
            while True:
                img, pre_img = self._capture_video()
                self._show_video(pre_img, img)

                contours, hi = cv2.findContours(
                    pre_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                amount = 0
                for contour in contours:
                    contour_area = cv2.contourArea(contour)

                    if contour_area > self._min_coin_area:
                        pass

        except KeyboardInterrupt:
            print(colored("\nExiting...", "red", attrs=["bold"]))
            exit(0)

        except Exception as e:
            print(e)
