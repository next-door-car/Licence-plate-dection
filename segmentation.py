import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from model.eval_svm_cn import SVM_CN
from model.eval_svm_gray import SVM_GRAY
from model.cut_level import*
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def Threshold(binary_image):  #对域值进行处理
#     # 计算垂直投影
#     vertical_projection = np.sum(binary_image, axis=0)
#     # 显示垂直投影直方图
#     plt.figure()
#     plt.plot(vertical_projection)
#     plt.title('Vertical Projection')
#     plt.show()

#     minima_indices = np.where((vertical_projection[:-2] > vertical_projection[1:-1]) &
#                           (vertical_projection[2:] > vertical_projection[1:-1]))[0] + 1

# #   打印每个极小值点的x轴坐标
# #     for index in minima_indices:
# #         x_coordinate = index
# #         print(f"zuobiao {x_coordinate}")
# # 获取极小值点的y轴坐标

#     y_coordinates = vertical_projection[minima_indices]
#     # threshold = np.median(y_coordinates)  # 中值阈值
#     threshold = np.mean(y_coordinates)      # 均值阈值
#     print(threshold)
#     class_0_indices = minima_indices[y_coordinates <= threshold]
#     class_1_indices = minima_indices[y_coordinates > threshold]
#     return minima_indices,y_coordinates , threshold ,class_0_indices

# def slice(character_images, binary_array):
#     lens = range(len(character_images) - 2)
#     classifier_CN = SVM_CN('PT/svmCn.dat')        
#     classifier_Gray = SVM_GRAY('PT/svmGray.dat')  # Create an instance of the SVM_Gray class
#     picture = []

#     for image_lens in lens:
#         start_x = character_images[image_lens]
#         end_x = character_images[image_lens + 1]
#         segmented_region = binary_array[:, start_x:end_x]
#         segmented_image = Image.fromarray(segmented_region * 255).convert('L')
#         segmented_image_array = np.array(segmented_image)
#         picture.append(segmented_image_array)
        

#         show(segmented_image_array)
#         cv2.imwrite(f'yc_picture/{image_lens}.png', segmented_image_array)

        

# def image_handle(binary_image):
#     binary_array = binary_image
#     selem = np.ones((3, 3), dtype=bool) #腐蚀大小
#     eroded_image = binary_erosion(binary_array, structure=selem)
#     dilated_image = binary_dilation(eroded_image, structure=selem)
#     filtered_image = Image.fromarray(dilated_image.astype(np.uint8) * 255)
#     filtered_array = np.array(filtered_image)
#     return filtered_array
#     # filtered_image.show()
# def draw_vertical_lines(image, x_coords):   #绘制图像直线
#     color_image = image.copy()
#     for x in x_coords:
#         cv2.line(color_image, (x, 0), (x, color_image.shape[0]), (0, 0, 255), 2)
#     return color_image

# def draw_edges_on_image(original_image, edges):
#     # 创建一个彩色图像用于绘制边缘
#     # color_image = cv2.cvtColor (original_image, cv2.COLOR_GRAY2BGR)
#     color_image=original_image
#     # 在边缘位置绘制红色曲线
#     edges = cv2.resize(edges, (color_image.shape[1], color_image.shape[0]))
#     color_image[edges != 0] = [0, 0, 255]  # 红色 (BGR格式)
#     return color_image

    
# def segment_characters(binary_image):

#     minima_indices ,y_coordinates,threshold ,class_0_indices =Threshold(binary_image)  #对域值进行处理
#     # 打印较小y轴坐标类别对应的极小值点的x轴坐标
#     X_count = []
#     for index in class_0_indices:
#         X_count.append(index)
#     if len(X_count)>9:
#         X_count = X_count[:9]  # 截取前8个元素
#     return X_count
      

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
    road = 'extracted_image.jpg'
    image = cv2.imread(road)
    image1 = cv2.imread(road)
    # image=chuli(image)
    # cv2.imwrite('extracted_image.jpg', images)
    images = scaling(image)   #再放缩图片
    gray_image = cv2.cvtColor (images, cv2.COLOR_BGR2GRAY)
    gray_image = two_value(gray_image) #二值化

    resized_image = cv2.resize(gray_image, (160, 40))   #尺度变换
    resized_image = 255 - resized_image
    
    # binary_array = resized_image
    #使用水平投影分割
    image2 = level(resized_image)  #会出现上下两峰
    show(image2)





    


    





