# import cv2
# import imutils
# from matplotlib import pyplot as pl
# import numpy as np
# import easyocr

import numpy as np
import matplotlib.pyplot as plt
import cv2

# read the image from arguments
image = cv2.imread('images/dashboard.png')

# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(grayscale, 10, 5)

# detect lines in the image using hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), 50, 5)
# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=2)

# show the image
plt.imshow(image)
plt.show()

# CODE FOR RECOGNIZE TEXT ON BIGGEST AREA;

# img = cv2.imread('images/dashboard.png')
# # img = cv2.GaussianBlur(img, (9, 9), 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # img = cv2.Canny(img, 90, 90)
#
# img_filter = cv2.bilateralFilter(gray, 11, 15, 15)
# edges = cv2.Canny(img_filter, 30, 200)
#
# cont = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cont = imutils.grab_contours(cont)
# cont = sorted(cont, key=cv2.contourArea, reverse=True)
#
# pos = None
#
# for c in cont:
#     approx = cv2.approxPolyDP(c, 10, True)
#     if len(approx) == 4:
#         pos = approx
#         break
#
# mask = np.zeros(gray.shape, np.uint8)
# new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
# bitwise_img = cv2.bitwise_and(img, img, mask=mask)
#
# (x, y) = np.where(mask == 255)
# (x1, y1) = np.min(x), np.min(y)
# (x2, y2) = np.max(x), np.max(y)
#
# crop = gray[x1:x2, y1:y2]
#
# text = easyocr.Reader(['en'])
# text = text.readtext(crop)
# print(text)
#
# # print(pos)
#
# pl.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
# pl.show()
