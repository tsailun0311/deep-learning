from __future__ import division, print_function
import os
from flask import Flask, request, render_template, jsonify, make_response
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd
import math
import base64

app = Flask(__name__)
mapp = pd.read_csv("static/emnist-byclass-mapping.txt",
                   delimiter=' ', index_col=0, header=None, squeeze=True)
device = torch.device("cpu")


class EMNIST_ResNet152(nn.Module):
    def __init__(self):
        super(EMNIST_ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, 62)

    def forward(self, x):
        logits = self.model(x)
        return logits


model = torch.jit.load('static/model.pth', map_location=device)
model.eval()
if not os.path.exists("uploads"):
    os.makedirs("uploads")


def imgf(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 降噪
    image = cv2.medianBlur(image, 5)   # 模糊化
    _, image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # OTSU二值化
    image = 255-image  # 反相
    image2 = cv2.Canny(image, 30, 150)  # 邊緣檢測
    kernel = np.ones((2, 2), np.uint8)
    image2 = cv2.dilate(image2, kernel, iterations=1)  # 膨脹
    contours, _ = cv2.findContours(
        image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        image = image[y:y+h, x:x+w]
        if w > h:
            new = math.ceil((w-h)/2)
            bordered_image = np.ones(
                (w+20, w+20), dtype=np.uint8) * 0  # 黑色背景
            bordered_image[new+10:new+h+10, 10:w+10] = image
            image = bordered_image
        else:
            new = math.ceil((h-w)/2)
            bordered_image = np.ones(
                (h+20, h+20), dtype=np.uint8) * 0
            bordered_image[10:h+10, new+10:new+w+10] = image
            image = bordered_image
    image = cv2.resize(image, (28, 28))
    image = cv2.dilate(image, kernel, iterations=1)  # 膨脹
    cv2.imwrite('static/css/1.png', image)
    image = cv2.flip(image, 1)  # 参数1表示水平翻转
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 順时针旋转90度
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.array(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        predictions = model(image)
        predicted_classes = predictions.argmax(dim=1)
    if max_contour is not None:
        result = chr(mapp[predicted_classes[0].item()])
    else:
        result = '失敗'
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_data = data["image"]
    if image_data.startswith("data:image/png;base64,"):
        image_data = image_data[len("data:image/png;base64,"):]
    image_binary = base64.b64decode(image_data)
    image_filename = './static/css/canva.png'
    with open(image_filename, 'wb') as f:
        f.write(image_binary)
    image = cv2.imread('./static/css/canva.png', cv2.IMREAD_GRAYSCALE)
    result = imgf(image)
    response = {
        'result': ('static/css/1.png', result)
    }
    return jsonify(response)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        result = imgf(image)
        response = {
            'result': ('static/css/1.png', result)
        }
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify(response)
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=12000)
