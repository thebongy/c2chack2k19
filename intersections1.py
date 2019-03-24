import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from math import sqrt
im = cv.imread('p1.png')
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

dst = cv.dilate(dst,None)
points = dst>0.01*dst.max()
points = np.array(list(zip(*(np.where(dst>0.01*dst.max())))))
unique = []


def dist(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

for point in points:
    for pnt in unique:
        if dist(point, pnt) < 10:
            break
    else:
        unique.append(point)

def find_extentions(pnt):
    print("Scanning surroundings of", pnt)
    tl = [pnt[1]-10, pnt[0]-10]
    br = [pnt[1]+10, pnt[0]+10]
    ext1 = None
    ext2 = None
    sol_h = None
    sol_v = None
    for i in range(21):
        h = [tl[0]+i, tl[1]]
        v = [tl[0], tl[1]+i]

        p = [br[0]-i, br[1]]
        q = [br[0], br[1]-i]
        try:
            if edges[h[1],h[0]] == 255:
                sol_h = tuple(h)
                # cv.circle(original, tuple(h), 2, [255,0,0])
            if edges[v[1],v[0]] == 255:
                sol_v = tuple(v)
                # cv.circle(original, tuple(v), 2, [0,255, 0])
            if edges[p[1],p[0]] == 255:
                sol_h = tuple(p)
                # cv.circle(original, tuple(p), 2, [0,255, 0])
            if edges[q[1],q[0]] == 255:
                sol_v = tuple(q)
                # cv.circle(original, tuple(q), 2, [0,255, 0])
        except Exception as e:
            print(e)
            pass
    cv.line(original, (pnt[1], pnt[0]), sol_h, [0, 255, 0], 1)
    cv.line(original, (pnt[1], pnt[0]), sol_v, [255, 0, 0], 1) 
for pnt in unique:
    cv.circle(original, (pnt[1], pnt[0]), 2, [255,255,0])
    find_extentions(pnt)
# def dist(p1, p2):
#     print(p1, p2)
#     return (p1[0]-p2[0])**2 - (p1[1]-p2[1])**2

# i = 0
# final = []
# while i<len(points):
#     current = points[i]
#     j = i+1
#     while j<len(points):
#         while j<len(points) and dist(current, points[j]) < 3:
#             points = np.delete(points, j,0)
#         j+=1
#     final.append(points[i])
#     i += 1

# print(final)
plt.subplot(221)
plt.imshow(original)
plt.subplot(222)
plt.xticks([])
plt.yticks([])
plt.imshow(blank)
plt.subplot(223)
plt.xticks([])
plt.yticks([])
plt.imshow(thresh, cmap="gray")
plt.subplot(224)
plt.xticks([])
plt.yticks([])
plt.imshow(edges, cmap="gray")
plt.show();