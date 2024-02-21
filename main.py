import warnings
warnings.filterwarnings("ignore")

import os
import json
from glob import glob
from tqdm import tqdm
import argparse

from src.predictor import Predictor

def point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def main(data_dir):
    image_paths = glob(os.path.join(data_dir, "*.jpg"))

    predictor = Predictor()
    
    true_answers = 0

    for image_path in tqdm(image_paths):
        bbox, image_id = predictor.predict(image_path)

        json_path = image_path.replace(".jpg", ".json")
        labelme_data = json.load(open(json_path))

        point = labelme_data["shapes"][0]["points"][0]
        true_answers += int(point_inside_bbox(point, bbox))

    print(f"Accuracy: {true_answers/len(image_paths)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for accuracy evaluation.")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing image and JSON files.")
    args = parser.parse_args()
    main(args.data_dir)
