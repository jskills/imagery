import cv2
import numpy as np
import matplotlib.pyplot as plt

### TODO
# directory or file as first arg
# options for conversion type : grayscale, edged, cartoon or all
# options for titles
# standard output directory
# adapt output file names based on filename being processed


# load and plot image
img = cv2.imread("./imagery-samples/0.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.savefig("i1.jpg")
plt.show()

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
plt.figure(figsize=(10,10))
plt.imshow(gray,cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.savefig("i2.jpg")
plt.show()

# get edged image
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
plt.figure(figsize=(10,10))
plt.imshow(edges,cmap="gray")
plt.axis("off")
plt.title("Edged Image")
plt.savefig("i3.jpg")
plt.show()

# cartoon time
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
plt.figure(figsize=(10,10))
plt.imshow(cartoon,cmap="gray")
plt.axis("off")
plt.title("Cartoon Image")
plt.savefig("i4.jpg")
plt.show()


