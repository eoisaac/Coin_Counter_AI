import cv2
import numpy as np
from termcolor import colored
from keras.models import load_model


class CoinCounter:
    def __init__(self):
        self._min_coin_area = 2000
        self.coin_classes = ["1 real", "50 cents", "25 cents"]

        self._model = load_model("./keras_model.h5", compile=False)
        self._data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

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

    def _detect_coins(self, area_img):
        coin_img = cv2.resize(area_img, (224, 224))
        coins_img = np.asarray(coin_img)

        normalized_coins_img = (coins_img.astype(np.float32) / 127.0) - 1
        self._data[0] = normalized_coins_img
        prediction = self._model.predict(self._data)
        index = np.argmax(prediction)
        percentage = prediction[0][index]
        coin_class = self.coin_classes[index]

        return coin_class, percentage

    def count_coins(self):
        print(colored("Counting coins...", "green", attrs=["bold"]))
        try:
            while True:
                img, pre_img = self._capture_video()

                contours, hi = cv2.findContours(
                    pre_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                amount = 0
                for contour in contours:
                    contour_area = cv2.contourArea(contour)

                    if contour_area > self._min_coin_area:
                        x, y, width, height = cv2.boundingRect(contour)
                        cv2.rectangle(
                            img, (x, y), (x + width, y + height), (0, 255, 0), 2)

                        area_img = img[y:y + height, x:x + width]
                        coin_class, percentage = self._detect_coins(area_img)

                        if percentage > 0.7:
                            cv2.putText(img, coin_class, (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            if coin_class == "1 real":
                                amount += 1
                            elif coin_class == "50 cents":
                                amount += 0.5
                            elif coin_class == "25 cents":
                                amount += 0.25

                # self._show_video(pre_img, img)
                cv2.rectangle(img, (430, 30), (600, 80), (0, 0, 0), -1)
                cv2.putText(img, f'R$ {amount}', (440, 67),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                cv2.imshow('IMG', img)
                cv2.imshow('IMG PRE', pre_img)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            print(colored("\nExiting...", "red", attrs=["bold"]))
            exit(0)

        except Exception as e:
            print(e)
