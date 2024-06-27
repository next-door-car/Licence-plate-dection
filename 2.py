import cv2 as cv
import cv2
import os   
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from model.eval_svm_cn import SVM_CN
from model.eval_svm_gray import SVM_GRAY
from model.cut_level import*
from model.cut_vertical import*
from sklearn.cluster import KMeans
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if not os.path.exists('yc_picture'):
    os.makedirs('yc_picture')

def scaling(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义HSV中蓝色的范围
    lower_blue = np.array([100, 150, 46])
    upper_blue = np.array([124, 255, 255])
    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # 将蓝色像素点替换为白色
    image[mask > 0] = [255, 255, 255]
    gray_image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    kerne2 = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kerne2) 
    #旋转特性 
    # show(closing)
    edges = cv2.Canny(closing, 100, 200)
    # show(edges)
    # 找到所有轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 将所有轮廓点合并到一个数组中
    all_points = np.vstack([contour for contour in contours])
    # 找到包含所有轮廓点的最小外接矩形
    x, y, w, h = cv2.boundingRect(all_points)
    # 直接使用边界矩形的坐标从原图像中截取对应的部分
    images = image1[y:y+h, x:x+w]
    # show(images)
    return images

def two_value(img):
    ret, img_thre = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 形态学处理:定义矩形结构
    closed = cv2.dilate(img_thre, kernel, iterations=1)  # 闭运算：迭代5次
    return closed


if __name__ == '__main__':
    # road = 'extracted_image.jpg'
    road ='picture\image1.png'
    image = cv2.imread(road)
    image1 = cv2.imread(road)
    # image=chuli(image)
    # cv2.imwrite('extracted_image.jpg', images)
    images = scaling(image)   #再放缩图片
    gray_image = cv2.cvtColor (images, cv2.COLOR_BGR2GRAY)
    gray_image = two_value(gray_image) #二值化

    resized_image = cv2.resize(gray_image, (160, 40))   #尺度变换
    resized_image = 255 - resized_image  # 反转灰度图

    #使用水平投影分割
    image2 = level(resized_image)  #会出现上下两峰

    show(image2)
    # pty = White_vertical(image2)
    pty = np.sum(image2 / 255, axis=0)  # Example vertical projection histogram

# Compute non-zero indices and mean values, starting from the third non-zero index
    non_zero_indices = np.where(pty > 0)[0]
    if non_zero_indices[0] == 0:
        non_zero_indices = non_zero_indices[1:]  # Remove the first zero index

    # Ensure starting from the third non-zero index
    if len(non_zero_indices) > 2:
        non_zero_indices = non_zero_indices[2:]

    mean_values = []
    values = []

    # Store all non-zero indices and their means
    print("All non-zero indices x:", non_zero_indices)
    mean_values = [non_zero_indices[0]]
    values =  [non_zero_indices[0]]
    print("Mean values for x indices with gap >= 3:")
    for i in range(len(non_zero_indices) - 1):
        if non_zero_indices[i + 1] - non_zero_indices[i] >= 3:
            mean_value = int((non_zero_indices[i] + non_zero_indices[i + 1]) / 2)
            mean_values.append(mean_value)
            values.append(non_zero_indices[i+1])

    if len(mean_values) != 9: 
        mean_max = values[-1]
        index_mean_max = np.where(non_zero_indices == mean_max)[0]
        # print(f"Mean_max = {mean_max} 在 non_zero_indices 中的索引值为: {index_mean_max[0]}")
        
        for i in range(index_mean_max[0], len(non_zero_indices) - 1):
            if non_zero_indices[i + 1] - non_zero_indices[i] >= 2:
                mean_value = int((non_zero_indices[i] + non_zero_indices[i + 1]) / 2)
                mean_values.append(mean_value)
    
    print(f'mean={mean_values}')

    for i in range(len(mean_values) - 1):
            x1 = mean_values[i]
            x2 = mean_values[i + 1]
            cropped_image = image2[:, x1:x2]  # Crop image between x1 and x2
        
            cv2.imwrite(f'yc_picture/{i}.png', cropped_image)  # 保存裁剪后的图片
                

    print("Stored mean values:", mean_values)

    # Plot vertical projection histogram
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(pty)), pty, color='black')
    plt.title('Vertical Projection Histogram')
    plt.xlabel('Column Index')
    plt.ylabel('Number of White Pixels')
    plt.show()




        



    


    





