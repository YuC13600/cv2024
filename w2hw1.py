import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
window_size = 15
c = 2
mean_image = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    window_size,
    c
)

ret, otsu_threshold = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Binary Image')
plt.imshow(mean_image, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Otsu Image')
plt.imshow(otsu_threshold, cmap='gray')
plt.show()

