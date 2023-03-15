import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('images/shapes.png')
img = cv2.imread('images/sign-in.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(255 - img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

potential_nested = list()
shapes = list()

for cnt, hry in zip(contours, hierarchy):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

    if len(approx) == 3:
        if hry[2] < 0:
            parent_idx = hry[3]
            parent_hier = hierarchy[parent_idx]
            if parent_hier[3] >= 0:
                potential_nested.append(cnt)
    else:
        shapes.append(cnt)

for nested in potential_nested:
    cv2.putText(img, "Nested", tuple(nested[0, 0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

for shape in shapes:
    cv2.putText(img, "Nested", tuple(shape[0, 0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

print(shapes)
plt.imshow(img)
plt.show()

