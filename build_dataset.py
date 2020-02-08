import math
import random

from cv2 import aruco
import matplotlib
import pandas as pd
import numpy as np
import cv2, PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.image as mpimg
# %matplotlib nbagg
# %matplotlib inline

# now the goal is to add perspective transform
def add_ar_tag_to_img(img):
    h = img.shape[0]
    w = img.shape[1]
    max_sz = min(h, w)*0.35
    id = 42 #random.randint(0, 999)
    mult = random.randint(1, int(max_sz/8))

    # create a random AR tag
    ar = cv2.aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_5X5_1000), id, sidePixels=8*8, img=img, borderBits=1)
    print(ar.shape)

    # fix shape
    ar_0 = np.zeros(shape=(ar.shape[0], ar.shape[1], 3))
    for i in range(3):
        ar_0[:, :, i] = ar

    # choose random center coordinate to place ar tag
    # c_x = random.randint(0,w)
    # c_y = random.randint(0,h)
    c_x = 100
    c_y = 200

    # Bounds on where AR tag will go in src img
    min_y = int(max(0, c_y-(ar.shape[0]/2)))
    min_x = int(max(0, c_x-(ar.shape[1]/2)))
    max_y = int(min(h, c_y+(ar.shape[0]/2)))
    max_x = int(min(w, c_x+(ar.shape[1]/2)))

    # Bounds on AR tag (i.e. how to cut the tag when it goes off the screen)
    min_y_ar = int(np.absolute(min(0,c_y-(ar.shape[0]/2))))
    min_x_ar = int(np.absolute(min(0,c_x-(ar.shape[1]/2))))
    max_y_ar = int(min(ar.shape[0],ar.shape[0]+(h-(c_y+(ar.shape[0]/2)))))
    max_x_ar = int(min(ar.shape[1],ar.shape[1]+(w-(c_x+(ar.shape[1]/2)))))

    selection = ar_0[min_y_ar:max_y_ar, min_x_ar:max_x_ar]
    plt.imshow(selection)
    plt.show()

    # pts for homography
    pts1 = np.float32([[min_y_ar, min_x_ar], [max_y_ar, min_x_ar], [min_y_ar, max_x_ar], [max_y_ar, max_x_ar]])
    pts2 = np.float32([
        [max_x-int(ar.shape[1]/2), min_y+int(ar.shape[0]/8)],
        [min_x, min_y],
        [max_x-int(ar.shape[1]/2), max_y-int(ar.shape[0]/8)],
        [min_x, max_y],
    ])
    # homography
    hom, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    ar_0_reg = cv2.warpPerspective(ar_0, hom, (w,h))
    plt.imshow(ar_0_reg)
    plt.show()


    mask2 = np.zeros_like(img, dtype=np.uint8)
    roi_corners = pts_to_roi(pts2)

    channel_count2 = img.shape[2]
    ignore_mask_color2 = (255,) * channel_count2

    cv2.fillConvexPoly(mask2, roi_corners, ignore_mask_color2)

    plt.imshow(mask2)
    plt.show()

    mask2 = cv2.bitwise_not(mask2)
    masked_image2 = cv2.bitwise_and(img, mask2)

    # Using Bitwise or to merge the two images
    final = cv2.bitwise_or(np.uint8(ar_0_reg), masked_image2)

    return final

# For some reason, the homography expects a certain ordering of the points
# that the roi mask doesn't expect thus we need to reorder the points.
def pts_to_roi(pts2):
    roi_corners = np.int32(pts2.copy())
    holder = roi_corners[2].copy()
    roi_corners[2] = roi_corners[3]
    roi_corners[3] = holder
    return roi_corners


def main():
    # pull in csv and read first few lines to figure out wtf you're supposed to do
    imgs = pd.read_csv("imgList.csv")
    img_name = imgs.loc[24][0]
    utah_desert = 'utah_desert'
    places = 'testSet_resize'
    # img_name = "utah_desert_1.jpg"


    img = cv2.imread('../{}/{}'.format(places, img_name),)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = add_ar_tag_to_img(img)

    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    main()







 # Translation matrix on the y axis Mat
# rotation = -1
# ry = np.array([
#     [math.cos(math.radians(rotation)), 0, -math.sin(math.radians(rotation))],
#     [0, 1, 0],
#     [math.sin(math.radians(rotation)), 0, math.cos(math.radians(rotation))],
# ])
# rx = np.array([
#     [1, 0, 0],
#     [0, math.cos(math.radians(rotation)), -math.sin(math.radians(rotation))],
#     [0, math.sin(math.radians(rotation)), math.cos(math.radians(rotation))],
# ])
# rz = np.array([
#     [math.cos(math.radians(rotation)), -math.sin(math.radians(rotation)), 0],
#     [math.sin(math.radians(rotation)), math.cos(math.radians(rotation)), 0],
#     [0, 0, 1],
# ])
# dst = np.zeros_like(ar_0)
#
# dst = cv2.warpPerspective(ar_0, ry, ar.shape)
# plt.imshow(dst)
# plt.show()