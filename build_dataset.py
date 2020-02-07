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


def add_ar_tag_to_img(img):
    h = img.shape[0]
    w = img.shape[1]
    max_sz = min(h, w)*0.35
    id = random.randint(0, 999)
    mult = random.randint(1, int(max_sz/8))

    # create a random AR tag
    ar = cv2.aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_5X5_1000), id, sidePixels=8*mult, img=img, borderBits=1)
    print(ar.shape)

    # fix shape
    ar_0 = np.zeros(shape=(ar.shape[0], ar.shape[1], 3))
    for i in range(3):
        ar_0[:, :, i] = ar

    # choose random center coordinate to place ar tage
    c_x = random.randint(0,w)
    c_y = random.randint(0,h)

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

    # put AR tag in src img
    selection = ar_0[min_y_ar:max_y_ar, min_x_ar:max_x_ar]
    img[min_y:max_y, min_x:max_x] = selection

    # Return ar's: presence (0|1), h, w, center (x,y)
    return img



def main():
    # pull in csv and read first few lines to figure out wtf you're supposed to do
    imgs = pd.read_csv("imgList.csv")
    img_name = imgs.loc[24][0]
    utah_desert = 'utah_desert'
    places = 'testSet_resize'
    img_name = "utah_desert_1.jpg"


    img = cv2.imread('../{}/{}'.format(utah_desert, img_name),)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = add_ar_tag_to_img(img)

    plt.imshow(img)
    plt.show()


def affine_test(src):
    srcTri = np.array([
        [0, 0],
        [src.shape[1] - 1, 0],
        [0, src.shape[0] - 1]
    ]).astype(np.float32)
    dstTri = np.array([
        [0, src.shape[1]*0.33],
        [src.shape[1] * 0.85, src.shape[0] * 0.25],
        [src.shape[1] * 0.15, src.shape[0] * 0.7]
    ]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
    # Rotating the image after Warp
    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = -50
    scale = 0.6
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
    # plt.imshow(src)
    # plt.show()
    plt.imshow(warp_dst)
    plt.show()
    # plt.imshow(warp_rotate_dst)
    # plt.show()


if __name__ == '__main__':
    main()
