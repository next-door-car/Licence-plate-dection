import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from model.eval_svm_cn import SVM_CN
from model.eval_svm_gray import SVM_GRAY
def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Threshold(binary_image):  #对域值进行处理
    # 计算垂直投影
    vertical_projection = np.sum(binary_image, axis=0)
    # 显示垂直投影直方图
    # plt.figure()
    # plt.plot(vertical_projection)
    # plt.title('Vertical Projection')
    # plt.show()

    minima_indices = np.where((vertical_projection[:-2] > vertical_projection[1:-1]) &
                          (vertical_projection[2:] > vertical_projection[1:-1]))[0] + 1

#   打印每个极小值点的x轴坐标
#     for index in minima_indices:
#         x_coordinate = index
#         print(f"zuobiao {x_coordinate}")
# 获取极小值点的y轴坐标

    y_coordinates = vertical_projection[minima_indices]
    # threshold = np.median(y_coordinates)  # 中值阈值
    threshold = np.mean(y_coordinates)      # 均值阈值
    print(threshold)
    class_0_indices = minima_indices[y_coordinates <= threshold]
    class_1_indices = minima_indices[y_coordinates > threshold]
    return minima_indices,y_coordinates , threshold ,class_0_indices

def slice(character_images, binary_array):
    lens = range(len(character_images) - 1)
    classifier_CN = SVM_CN('PT/svmCn.dat')        
    classifier_Gray = SVM_GRAY('PT/svmGray.dat')  # Create an instance of the SVM_Gray class
    picture = []

    for image_lens in lens:
        start_x = character_images[image_lens]
        end_x = character_images[image_lens + 1]
        segmented_region = binary_array[:, start_x:end_x]
        segmented_image = Image.fromarray(segmented_region * 255).convert('L')
        segmented_image_array = np.array(segmented_image)
        picture.append(segmented_image_array)
        
        # Display or save the segmented image if needed
        # show(segmented_image_array)
        # cv2.imwrite(f'yc_picture/{image_lens}.png', segmented_image_array)

        # Predict the class using classifier_CN and classifier_Gray
        # predicted_class_CN = classifier_CN.predict(segmented_image_array)
        # predicted_class_Gray = classifier_Gray.predict(segmented_image_array)
        # print(f'Segment {image_lens} Predicted class by CN: {predicted_class_CN}, by Gray: {predicted_class_Gray}')
    print(picture)

def image_handle(binary_image):
    binary_array = binary_image
    selem = np.ones((3, 3), dtype=bool) #腐蚀大小
    eroded_image = binary_erosion(binary_array, structure=selem)
    dilated_image = binary_dilation(eroded_image, structure=selem)
    filtered_image = Image.fromarray(dilated_image.astype(np.uint8) * 255)
    filtered_array = np.array(filtered_image)
    return filtered_array
    # filtered_image.show()
def draw_vertical_lines(image, x_coords):   #绘制图像直线
    color_image = image.copy()
    for x in x_coords:
        cv2.line(color_image, (x, 0), (x, color_image.shape[0]), (0, 0, 255), 2)
    return color_image

def draw_edges_on_image(original_image, edges):
    # 创建一个彩色图像用于绘制边缘
    # color_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    color_image=original_image
    # 在边缘位置绘制红色曲线
    edges = cv2.resize(edges, (color_image.shape[1], color_image.shape[0]))
    color_image[edges != 0] = [0, 0, 255]  # 红色 (BGR格式)
    return color_image

    
def segment_characters(binary_image):

    minima_indices ,y_coordinates,threshold ,class_0_indices =Threshold(binary_image)  #对域值进行处理
    # 打印较小y轴坐标类别对应的极小值点的x轴坐标
    X_count = []
    for index in class_0_indices:
        X_count.append(index)
    return X_count
      



if __name__ == '__main__':
    road = 'extracted_image.jpg'
    image = cv2.imread(road)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray_image, (160, 40))   #尺度变换
    resized_image = cv2.GaussianBlur(resized_image, (3, 3), 0) 
    normalized_image = (resized_image - np.min(resized_image)) * (100 / (np.max(resized_image) - np.min(resized_image)))

    image_success= image_handle(normalized_image)   #<class 'PIL.Image.Image'>
    # 将灰度值前 20% 的点的灰度值乘以 2.55
    threshold = np.percentile(image_success, 20)
    processed_image = np.where(normalized_image <= threshold, normalized_image * 2.55, normalized_image)
    binary_array = processed_image.astype(np.uint8)
    # edges = cv2.Canny(binary_array, 50, 150, apertureSize=3)  #边缘检测
    character_images = segment_characters(binary_array)
    print(character_images)
    slice(character_images,binary_array)
    image_with_lines = draw_vertical_lines(binary_array, character_images)
    
    
    # # 显示结果图像
    # plt.figure(figsize=(10, 5))
    # plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Red Vertical Lines')
    # plt.axis('off')
    # plt.show()



    # result_image = draw_edges_on_image(image, edges)
    # plt.figure(figsize=(10, 5))
    # plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    # plt.title('Edges Detected and Highlighted')
    # plt.axis('off')
    # plt.show()

    

