import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 统计白色像素点（分别统计每一列）
def White_Statistic(image):
    pty = []  # 每列白色像素个数
    height, width = image.shape
    # 逐列遍历
    for i in range(width):
        num = 0
        for j in range(height):
            if (image[j][i] == 255):
                num += 1
        pty.append(num)
    return pty

# 纵向分割：分割字符
def Cut_Y(pty, cols, h1, h2, binary):
    WIDTH = 32          # 经过测试，一个字符宽度约为32
    w1 = w2 = 0         # 字符开始和结束
    begin = False       # 字符开始标记
    last = 10           # 上一次的值
    con = 0             # 计数

    for j in range(cols):
        if pty[j] == max(pty):
            if j < 30 or j > 270:
                if begin:
                    begin = False
                continue
            if begin:
                begin = False
                b_copy = binary[h1:h2, w1:w2]
                cv.imshow(f'binary-{con}', b_copy)
                cv.imwrite(f'car_characters/image-{con}.jpg', b_copy)
                con += 1
                break

        if pty[j] < 12 and not begin:
            last = pty[j]
        elif last < 12 and pty[j] > 20:
            last = pty[j]
            w1 = j
            begin = True
        elif pty[j] < 13 and begin:
            begin = False
            last = pty[j]
            w2 = j
            width = w2 - w1
            if 10 < width < WIDTH + 3:
                b_copy = binary[h1:h2, w1:w2]
                cv.imshow(f'binary-{con}', b_copy)
                cv.imwrite(f'car_characters/image-{con}.jpg', b_copy)
                con += 1
            elif width >= WIDTH + 3:
                num = int(width / WIDTH + 0.5)
                for k in range(num):
                    w3 = w1 + k * WIDTH
                    w4 = w1 + (k + 1) * WIDTH
                    b_copy = binary[h1:h2, w3:w4]
                    cv.imshow(f'binary-{con}', b_copy)
                    cv.imwrite(f'car_characters/image-{con}.jpg', b_copy)
                    con += 1

# 分割车牌图像（根据直方图）
def Cut_Image(pty, binary, h1, h2):
    rows, cols = binary.shape
    Cut_Y(pty, cols, h1, h2, binary)

# 中值滤波并灰度化处理
def preprocess_image(image):
    mid = cv.medianBlur(image, 5)
    gray = cv.cvtColor(mid, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    return binary

# 使用垂直投影分割字符
def segment_characters(image, h1, h2):
    binary = preprocess_image(image)
    pty = White_Statistic(binary)
    Cut_Image(pty, binary, h1, h2)

# 示例使用
if __name__ == '__main__':
    image_path = 'path_to_license_plate_image.jpg'
    image = cv.imread(image_path)
    h1, h2 = 10, 50  # 假设的上下边界，需要根据实际情况调整
    segment_characters(image, h1, h2)
    cv.waitKey(0)
    cv.destroyAllWindows()
