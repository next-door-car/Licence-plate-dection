import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
import matplotlib.pyplot as plt

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 二-5、统计白色像素点（分别统计每一行、每一列）
def White_Statistic(image):
    ptx = []  # 每行白色像素个数
    height, width = image.shape
    # 逐行遍历
    for i in range(height):
        num = 0
        for j in range(width):
            if(image[i][j]==255):
                num = num+1
        ptx.append(num)
 
    return ptx
# 二-7-2、横向分割：上下边框

def Cut_level(ptx, rows):
    # 横向切割（分为上下两张图，分别找其波谷，确定顶和底）
    # 1、下半图波谷
    min, r = 300, 0
    for i in range(int(rows / 2)):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h1 = r  # 添加下行（作为顶）
    # 2、上半图波谷
    min, r = 300, 0
    for i in range(int(rows / 2), rows):
        if ptx[i] < min:
            min = ptx[i]
            r = i
    h2 = r  # 添加上行（作为底）
    return h1, h2

def level(image):
    # 5、统计各行各列白色像素个数（为了得到直方图横纵坐标）
    ptx = White_Statistic(image)
    rows = len(ptx)
    row = [i for i in range(rows)]
    # 横向直方图
    plt.barh(row, ptx, color='black', height=1)
    plt.show()
    h1, h2 = Cut_level(ptx, rows)
    cut_level = image[h1:h2, :]
    return cut_level