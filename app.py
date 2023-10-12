from __future__ import division, print_function
import os
from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
mapp = pd.read_csv("static/emnist-byclass-mapping.txt",
                   delimiter=' ', index_col=0, header=None, squeeze=True)
device = torch.device("cuda")


class EMNIST_ResNet152(nn.Module):
    def __init__(self):
        super(EMNIST_ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, 62)

    def forward(self, x):
        logits = self.model(x)
        return logits


model = torch.load('static/model.pth', map_location=device)
model.eval()
if not os.path.exists("uploads"):
    os.makedirs("uploads")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (28, 28))
        image = cv2.flip(image, 1)  # 参数1表示水平翻转
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 順时针旋转90度
        image = 255-image
        image = np.array(image)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            predictions = model(image)
            predicted_classes = predictions.argmax(dim=1)
        result = chr(mapp[predicted_classes[0].item()])
        if os.path.exists(file_path):
            os.remove(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
