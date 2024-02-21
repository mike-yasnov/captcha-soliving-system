import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

from src.utils.paths import PATH_TO_YOLO_MODEL
from src.utils.convertrer import convert_yolov8_bbox_to_opencv
from src.utils.preprocess import preprocess_image
from src.match import BboxCircleMatcher
from src.image_similarity import transformation_chain, extract_embeddings, fetch_similar

class Predictor:
    """
    Captcha solver class
    """
    def __init__(self, ):
        self.__yolo_model = YOLO(PATH_TO_YOLO_MODEL, verbose=False)
        self.__matcher = BboxCircleMatcher()

        self.circles = {
            "left": [
                {"id": 1, "center": (50, 190), "radius": 74},
                {"id": 2, "center": (60, 190), "radius": 97},
                {"id": 3, "center": (70, 190), "radius": 127},
                {"id": 4, "center": (75, 190), "radius": 153},
                {"id": 5, "center": (80, 190), "radius": 180}
            ],
            "right": [
                {"id": 1, "center": (150, 190), "radius": 74},
                {"id": 2, "center": (140, 190), "radius": 97},
                {"id": 3, "center": (130, 190), "radius": 127},
                {"id": 4, "center": (125, 190), "radius": 153},
                {"id": 5, "center": (120, 190), "radius": 180}
            ]
        }


    def get_sun_center(self, image: np.array):
        """
        Get sun center coordinates

        @param image - captcha image 
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Определение диапазона желтого цвета в HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Создание маски, которая оставляет только желтые пиксели
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Нахождение контуров на изображении
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threshold_area = 2000
        center = (0, 0)
        # Итерация по всем найденным контурам
        for contour in contours:
            # Вычисление параметров контура
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # Вычисление координат центра контура
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                # Проверка на то, что область контура достаточно большая
                if area > threshold_area:
                    center = (cx, cy)  # Рисуем круг в центре контура
        return center 
    

    def __detect(self, image: np.array):
        """
        After prediction we get image with shape (400, 200)
        This function splits image on two (200, 200) and match bboxes for both of them 

        @param prediction - Yolo v8 prediction 
        """

        prediction = self.__yolo_model(image, verbose=False)[0]
        height, width = prediction.orig_shape

        bboxes = [convert_yolov8_bbox_to_opencv(x, width, height) for x in prediction.boxes.xywhn]
        classes = prediction.boxes.cls

        icons_bboxes, number = [], 1

        for (bbox, cls) in zip(bboxes, classes):
            _, y_center = self.__matcher.bbox_center(bbox)
            if y_center > 200:
                if cls != 5:
                    number = int(cls)+1
                else:
                    x1, y1, x2, y2 = bbox
                    icon = image[y1:y2, x1:x2]
            else:
                if cls == 5:
                    icons_bboxes.append(bbox)

        return icons_bboxes, number, icon
    
    def predict(self, image_path: str) -> list:
        """
        Main prediction function

        @param image_path - path to image
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_for_prediction = image[:, :200, :]

        captchas = [image[:200, i:i+200, :] for i in range(0, 1000, 200)]
        bboxes, number, icon = self.__detect(image_for_prediction)

        sun_x, sun_y = self.get_sun_center(captchas[0])
        circles = self.circles["left"] if sun_x < 100 else self.circles["right"]
        bbox = self.__matcher.match(circles, bboxes)[number]

        x1, y1, x2, y2 = bbox
        captcha_icons = [x[y1:y2, x1:x2] for x in captchas]
        captcha_icons_transformed = [preprocess_image(captcha_icon) for captcha_icon in captcha_icons]
        captcha_icons_transformed = [transformation_chain(Image.fromarray(captcha_icon)).unsqueeze(0) for captcha_icon in captcha_icons_transformed]
        captcha_icons_transformed = torch.cat(captcha_icons_transformed, dim=0)

        icon_transformed = transformation_chain(Image.fromarray(icon)).unsqueeze(0)
        embeddings = extract_embeddings(captcha_icons_transformed)

        image_id = fetch_similar(icon_transformed, embeddings, list(range(len(embeddings)))) 
        x1, x2 = x1 + 200 * image_id, x2 + 200 * image_id
        return [x1, y1, x2, y2], image_id

