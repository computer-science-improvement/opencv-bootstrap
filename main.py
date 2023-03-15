import cv2
import numpy as np
import skimage.filters as filters

kernel = np.ones((5, 5), np.uint8)

# img = cv2.imread('./images/maxresdefault.jpg')
img = cv2.imread('./images/sign-in.jpg')
# img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
# img = cv2.GaussianBlur(img, (1, 1), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(img, (95,95), 0)
division = cv2.divide(img, smooth, scale=255)
sharp = filters.unsharp_mask(division, radius=100, amount=100, preserve_range=False)
sharp = (255*sharp).clip(0,255).astype(np.uint8)

thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# part of image
# img = img[0:100, 0:200]

# img = cv2.Canny(img, 50, 50)
# img = cv2.dilate(img, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)

cv2.imshow('thresh', thresh)

# cv2.imshow('Result', img)


cv2.waitKey(0)