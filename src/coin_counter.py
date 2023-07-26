import os
from time import sleep
import cv2
import numpy as np
from termcolor import colored
from keras.models import load_model

colors = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
}


class CoinCounter:
    def __init__(self):
        self._min_coin_area = 2000
        self.coins = {
            "1_real": {
                "class": "1_real",
                "label": "1 real",
                "value": 1,
            },
            "50_cents": {
                "class": "50_cents",
                "label": "50 centavos",
                "value": 0.5,
            },
            "25_cents": {
                "class": "25_cents",
                "label": "25 centavos",
                "value": 0.25,
            },
            "10_cents": {
                "class": "10_cents",
                "label": "10 centavos",
                "value": 0.1,
            },
            "5_cents": {
                "class": "5_cents",
                "label": "5 centavos",
                "value": 0.05,
            },
        }

        self._model = load_model("src/model/keras_model.h5", compile=False)
        self._data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        self._video = cv2.VideoCapture(0)

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

    def _detect_coin(self, area_img):
        classes_list = [class_ for class_ in self.coins.keys()]
        coin_img = cv2.resize(area_img, (224, 224))
        coins_img = np.asarray(coin_img)

        normalized_coins_img = (coins_img.astype(np.float32) / 127.0) - 1
        self._data[0] = normalized_coins_img
        prediction = self._model.predict(self._data)
        index = np.argmax(prediction)
        percentage = prediction[0][index]
        coin_class = classes_list[index]

        return coin_class, percentage

    def save_coins_img(self):
        path = os.path.join(os.getcwd(), "src", "images")
        if not os.path.exists(path):
            os.mkdir(path)

        for coin_name, coin in self.coins.items():
            try:
                coin_class = coin.get('class')
                coin_label = coin.get('label')
                coin_value = coin.get('value')

                print(
                    colored(f"Insert {coin_label} coin...", "yellow", attrs=["bold"]))
                print(colored("Start capturing? (y/n)",
                      "yellow", attrs=["bold"]))
                capture = input(">>> ")

                if capture == "y":
                    coin_img_count = 0
                    coin_folder_path = os.path.join(path, coin_class)
                    if not os.path.exists(coin_folder_path):
                        os.mkdir(coin_folder_path)

                    while coin_img_count < 100:
                        img, pre_img = self._capture_video()

                        img_filename = f"{coin_class}_{coin_img_count + 1}.jpg"
                        img_filepath = os.path.join(
                            coin_folder_path, img_filename)

                        cv2.imshow(f"{coin_label} Coin", img)
                        cv2.imwrite(img_filepath, img)

                        coin_img_count += 1
                        print(
                            f"Captured {coin_img_count} images for {coin_label} coin.")
                        sleep(0.1)

            except KeyboardInterrupt:
                print(colored("\nExiting...", "red", attrs=["bold"]))
                exit(0)

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
                            img, (x, y), (x + width, y + height), colors.get("green"), 2)

                        area_img = img[y:y + height, x:x + width]
                        coin_class, percentage = self._detect_coin(area_img)

                        if percentage > 0.7:
                            cv2.putText(img, coin_class, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get("green"), 1)

                            amount += self.coins.get(coin_class).get("value")

                cv2.rectangle(img, (430, 30), (600, 80),
                              colors.get("black"), -1)
                cv2.putText(img, f'R$ {amount}', (440, 67),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors.get("white"), 2)

                self._show_video(pre_img, img)

        except KeyboardInterrupt:
            print(colored("\nExiting...", "red", attrs=["bold"]))
            exit(0)

        except Exception as e:
            print(e)
