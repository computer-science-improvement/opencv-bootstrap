import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
image = cv2.imread('./images/sign_up .png')
#
# image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(image, 250, 250, 0)

print(pytesseract.image_to_string(threshold_image))

plt.imshow(threshold_image)
plt.show()
