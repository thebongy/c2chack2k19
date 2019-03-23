import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
im = cv.imread('p3.png')
original = im.copy()
im[np.where((im==[156,222,251]).all(axis=2))] = [255,255,255]
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 250, 255, cv.THRESH_BINARY)
contours, _test= cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda x:cv.contourArea(x)>=250, contours))
blank = np.ones(im.shape)
cv.drawContours(blank, contours, -1, (0,255,0), 1)
edges = cv.Canny(np.uint8(blank),100,200)

dst = cv.cornerHarris(thresh,2,5,0.05)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
edges = dst>0.01*dst.max()
points=np.unravel_index(dst.argmax(),dst.shape)
print(list(points))
plt.subplot(141)
plt.imshow(original)
plt.subplot(142)
plt.imshow(blank)
plt.subplot(143)
plt.imshow(thresh, cmap="gray")
plt.subplot(144)
plt.imshow(edges, cmap="gray")
plt.show();