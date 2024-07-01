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

def angle(image):
    median_image = cv2.medianBlur(image, 5)  # 5是滤波器的大小，必须为奇数
    median_image = 255-median_image
    # 膨胀操作
    kerne3 = np.ones((30, 30), np.uint8)
    dilated_image = cv2.dilate(median_image, kerne3, iterations=1)    #闭操作
    show(dilated_image)

    # Canny 边缘检测
    edges = cv2.Canny(median_image, 70, 200, apertureSize=5, L2gradient=False)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    # 创建一个彩色图像用于绘制检测到的直线
    line_image = cv2.cvtColor(median_image, cv2.COLOR_GRAY2BGR)
    # show(line_image)
    # 检测到的直线角度
    angles = []
    # 绘制检测到的直线
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

    # show(line_image)
    # 删除最大值与最小值
    angles_array = np.array(angles)
    angles_array = angles_array[angles_array != 0.0]  # 过滤掉值为 0.0 的元素
    if len(angles_array) > 2: #长度大于2才删除
        min_value = np.min(angles_array)
        max_value = np.max(angles_array)
        # 通过逻辑运算符 & 和 | 处理最小值和最大值删除
        # 找到最小值和最大值的索引
        min_indices = np.where(angles_array == min_value)[0]
        max_indices = np.where(angles_array == max_value)[0]
        # 如果最小值出现超过一次，只删除一个
        if len(min_indices) > 1:
            angles_array = np.delete(angles_array, min_indices[0])
        else:
            angles_array = angles_array[(angles_array != min_value)]
        # 如果最大值出现超过一次，只删除一个
        if len(max_indices) > 1:
            angles_array = np.delete(angles_array, max_indices[0]) 
        else:
            angles_array = angles_array[(angles_array != max_value)]
       

    if len(angles_array) == 0:
        initial_average =0
        filtered_angles_array = angles_array
    else:
        initial_average = np.mean(angles_array)
        filtered_angles_array = angles_array[np.abs(angles_array - initial_average) <= 10]#与均值相差10就保留
    
    if angles:
        if len(filtered_angles_array) == 0:
            average_angle = initial_average
        else:
            average_angle = np.mean(filtered_angles_array)
        print(f'Average Angle: {average_angle}')
    else:
        average_angle = 0
        print("No lines detected with average_angle.")
    return average_angle
def scaling(image):
    image_cope = image.copy()
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义HSV中蓝色的范围
    lower_blue = np.array([100, 150, 46])
    upper_blue = np.array([124, 255, 255])
    # 创建掩码
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # 将蓝色像素点替换为白色
    image[mask > 0] = [255, 255, 255]
    gray_image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY_INV)
    kerne2 = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kerne2) 
    
    # show(images)

    average_angle = angle(closing)
    if average_angle !=0:
        # 获取图像的尺寸
        (h, w) = closing.shape[:2]
        # 获取图像中心点
        center = (w // 2, h // 2)
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, average_angle, 1.0)
        # 进行仿射变换（旋转图像）
        rotated_closing = cv2.warpAffine(closing, rotation_matrix, (w, h))
        show(rotated_closing)
        if np.all(rotated_closing == 255): #如果图片全白(中值滤波时出现的问题)，则返回原图
            images = image_cope
        else:
            #旋转特性 
            # show(closing)
            edges = cv2.Canny(rotated_closing, 100, 200)
            show(edges)
            # 找到所有轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 将所有轮廓点合并到一个数组中
            all_points = np.vstack([contour for contour in contours])
            # 找到包含所有轮廓点的最小外接矩形
            x, y, w, h = cv2.boundingRect(all_points)
            # 直接使用边界矩形的坐标从原图像中截取对应的部分
            images = image1[y:y+h, x:x+w]
            if x==0 and y==0:  #说明没有进行截取操作
                images = cv2.warpAffine(image1, rotation_matrix, (w, h))
    else:
        images = image_cope
    show(images)    
    return images

def two_value(img):
    ret, img_thre = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  
    closed = cv2.dilate(img_thre, kernel, iterations=1)  # 闭运算：迭代5次
    return closed


if __name__ == '__main__':
    road ='extracted_image.jpg'
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
    # show(image2)
    # 高斯滤波
    image2 = cv2.GaussianBlur(image2, (3, 3), 0)
    show(image2)
    pty = np.sum(image2 / 255, axis=0)  


    #以下会出现两种情况
    #1.汉字前无干扰
    #2.汉字前有干扰
    average = sum(pty[:2]) / 2 #前两个pty是否都为0，判断前方是否有干扰
    non_zero_indice = np.where(pty > 0)[0]
    non_zero_indices = np.where(pty > 2)[0]
    mean_values = []
    mean_values.append(0)  # 第一位进行占位
    values = []
    FLAG = 0
    if non_zero_indice[0] == 0: # ty分量大于0
            non_zero_indice = non_zero_indice[1:]  
    if non_zero_indices[0] == 0:# ty分量大于3
            non_zero_indices = non_zero_indices[1:]  
    if len(non_zero_indices) > 2:
            non_zero_indices = non_zero_indices[2:]


    # Store all non-zero indices and their means
    
    # Store all non-zero indices and their means
    print("All non-zero indices x:", non_zero_indices)
    if average == 0:   #前方两个pty都为0，无干扰，mean_values取非0的第一个值
        FLAG=1
        mean_values[0] = (non_zero_indice[0])
    else:   #有干扰  mean_values取非0的第一个值      
            #在10之内找到pty合适的点
            for i in range(non_zero_indice[0],9): #无0
                 if(pty[i] ==0):
                      FLAG=1
                      mean_values[0]=i
            if len(mean_values) == 0: #说明在non_zero_indices[0]和10之间没有pty = 0的点（无0,1）
                for j in range(non_zero_indice[0],9):
                  if(pty[j] < 2):
                      FLAG=1
                      mean_values[0]= j
            if len(mean_values) == 0: #说明在non_zero_indices[0]和10之间没有pty = 0的点(无0,1,2)
                for t in range(non_zero_indice[0],9):
                  if(pty[t] < 3):
                      FLAG=1
                      mean_values[0]=t
            #说明在10 之内所有pty均大于3
            if len(mean_values) == 0: #说明图像处理时出现了问题,手动将mean_values赋值为0-10内最小值索引
                print('picture one have a problem')
                 # 获取索引范围 0 到 9 内的子数组
                sub_array = pty[:10]
                # 找到最小值及其索引
                min_value = min(sub_array)
                FLAG=1
                mean_values[0]= sub_array.index(min_value)
    values =  [mean_values[0]]     

    #范围从values到 non_zero_indices的末尾
    print("Mean values for x indices with gap > 3:")
    for i in range(len(non_zero_indices) - 1):
        if non_zero_indices[i + 1] - non_zero_indices[i] >= 2:
            mean_value = int((non_zero_indices[i] + non_zero_indices[i + 1]) / 2)
            if FLAG ==1: ##说明mean_value 第一位已经赋值
                FLAG=0    
                # if mean_value > mean_values[0]:
                # mean_values[0]=mean_value
            else:
                mean_values.append(mean_value)
            values.append(non_zero_indices[i+1])
    if len(mean_values)>2 and len(values)>2:
        if mean_values[1] - mean_values[0]>=35: #数字与汉字之间少了一个
            if mean_values[1] - values[1]>10: #加入限定条件
                mean_values.insert(1, values[1])  # 在第一位插入values[1]



    if len(mean_values) < 8: 
        # mean_max = values[-1]
        # index_mean_max = np.where(non_zero_indices == mean_max)[0]
        # print(f"Mean_max = {mean_max} 在 non_zero_indices 中的索引值为: {index_mean_max[0]}")
        # for i in range(index_mean_max[0], len(non_zero_indice) - 1):
            # if non_zero_indices[i + 1] - non_zero_indices[i] >= 2:
            #     mean_value = int((non_zero_indices[i] + non_zero_indices[i + 1]) / 2)
        mean_values.append(158)
    if mean_values[0] <10 and mean_values[1] < 10:
        mean_values.pop(0)
    print(f'mean={mean_values}')

    for i in range(len(mean_values) - 1):
            x1 = mean_values[i]
            x2 = mean_values[i + 1]
            cropped_image = image2[:, x1:x2]  
            cv2.imwrite(f'yc_picture/{i}.png', cropped_image)  # 保存裁剪后的图片
                

    print("Stored mean values:", mean_values)

    # Plot vertical projection histogram
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(pty)), pty, color='black')
    plt.title('Vertical Projection Histogram')
    plt.xlabel('Column Index')
    plt.ylabel('Number of White Pixels')
    plt.show()




        



    


    





