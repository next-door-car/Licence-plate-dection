import cv2 as cv
import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
import matplotlib.pyplot as plt

def White_vertical(image):
    pty = []  # 每列白色像素个数
    height, width = image.shape
    # 逐列遍历
    for i in range(width):
        num = 0
        for j in range(height):
            if (image[j][i] == 255):
                num = num + 1
        pty.append(num)
 
    return pty

# 二-6、绘制直方图
def Draw_Hist(pty):
    # 依次得到各行、列
    cols = len(pty)
    col = [j for j in range(cols)]
    # 纵向直方图
    plt.bar(col, pty, color='black', width=1)
    #       横    纵
    plt.show()

