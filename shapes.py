import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./images/sign_up .png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

HAS_SHAPE = {
    "T": False,
    "Q": False,
    "P": False,
    "H": False,
    "C": False
}

contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(image, contour, 0, (0, 0, 0), 4)
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + w / 3)
    y_mid = int(y + h / 1.5)

    coords = (x_mid, y_mid)
    colours = (0, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX

    if len(approx) == 3 and HAS_SHAPE['T']:
        cv2.putText(image, 'Triangle', coords, font, 1, colours, 1)
        HAS_SHAPE['T'] = True
    if len(approx) == 4:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        cv2.putText(image, 'Quadrilateral', coords, font, 1, colours, 1)
        HAS_SHAPE['Q'] = True
    if len(approx) == 5 and HAS_SHAPE['P']:
        cv2.putText(image, 'Pentagon', coords, font, 1, colours, 1)
        HAS_SHAPE['P'] = True
    if len(approx) == 6 and HAS_SHAPE['H']:
        cv2.putText(image, 'Hexagon', coords, font, 1, colours, 1)
        HAS_SHAPE['P'] = True
    else:
        if not HAS_SHAPE['C']:
            cv2.putText(image, 'Circle', coords, font, 1, colours, 1)
            HAS_SHAPE['C'] = True


plt.imshow(image)
plt.show()

