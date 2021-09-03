import numpy as np
import requests
import json
import cv2
import base64
import os

def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

if __name__ == "__main__":
    url = "http://127.0.0.1:18080/imagenet/prediction"
    with open(os.path.join(".", "daisy.jpg"), 'rb') as file:
        image_data1 = file.read()
    image = cv2_to_base64(image_data1)
    data = {"key": ["image"], "value": [image]}
    for i in range(100):
        r = requests.post(url=url, data=json.dumps(data))
        print(r.json())
