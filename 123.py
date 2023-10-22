import cv2
import math
import numpy as np
image = cv2.imread('static/css/4.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image, (5, 5), 0)  # 降噪
image = cv2.medianBlur(image, 5)   # 模糊化
cv2.imwrite('out1.jpg', image)
_, image = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU二值化
image = 255-image  # 反相
cv2.imwrite('out2.jpg', image)
image2 = cv2.Canny(image, 30, 30)  # 邊緣檢測
cv2.imwrite('out3.jpg', image2)
kernel = np.ones((2, 2), np.uint8)
image2 = cv2.dilate(image2, kernel, iterations=9)  # 再膨脹
cv2.imwrite('out4.jpg', image2)
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
    cv2.imwrite('out5.jpg', cv2.rectangle(
        image, (x, y), (x + w, y + h), (255, 255, 255), 2))
    cv2.rectangle(
        image, (x, y), (x + w, y + h), (0, 0, 0), 2)
    image = image[y:y+h, x:x+w]
    cv2.imwrite('out6.jpg', image)
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
cv2.imwrite('out7.jpg', image)
