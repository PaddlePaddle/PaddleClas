import requests
import json
import base64
import os

imgpath = "../../drink_dataset_v1.0/test_images/001.jpeg"

def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

if __name__ == "__main__":
    url = "http://127.0.0.1:18081/recognition/prediction"

    with open(os.path.join(".",  imgpath), 'rb') as file:
        image_data1 = file.read()
    image = cv2_to_base64(image_data1)
    data = {"key": ["image"], "value": [image]}

    for i in range(1):
        r = requests.post(url=url, data=json.dumps(data))
        print(r.json())
