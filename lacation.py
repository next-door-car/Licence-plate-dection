import cv2
import numpy as np
from matplotlib import pyplot as plt

    

if __name__ == '__main__':
    image_path = 'example/1.jpg'
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([100, 100, 46])
    upper_bound = np.array([124, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    output_image1 = np.ones_like(image) * 255
    output_image1[mask == 255] = [0, 0, 0]

    gaussian_blur = cv2.GaussianBlur(output_image1, (5, 5), 0)
    median_blur = cv2.medianBlur(gaussian_blur, 5)
    gray_image = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)  #开操作
    median_blur = cv2.medianBlur(opening, 7)

    kerne2 = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(median_blur, cv2.MORPH_CLOSE, kerne2) 

    kerne3 = np.ones((20, 20), np.uint8)
    closing = cv2.dilate(closing, kerne3, iterations=1)  
    # show(closing)

    edges = cv2.Canny(closing, 70, 200, apertureSize=5, L2gradient=False)
    # show(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    # 创建一个彩色图像用于绘制检测到的直线
    line_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
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

     # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的旋转外接矩形
    max_area = 0
    best_rect = None
    best_box = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        area = rect[1][0] * rect[1][1]
        if area > max_area: 
            max_area = area
            best_rect = rect
            best_box = box

    print("最大矩形面积：", max_area)
    if best_rect is not None:
        # 计算当前矩形的宽高
        current_width, current_height = best_rect[1]
        print("当前矩形宽高：", current_width, current_height)
        # 判断并调整矩形的长和宽
        width1 = current_width
        height1 = current_height
        long = None
        short = None
        # 创建新的旋转矩形
        expanded_rect = ((best_rect[0][0], best_rect[0][1]), (current_width, current_height), best_rect[2])
        expanded_box = cv2.boxPoints(expanded_rect)
        expanded_box = np.intp(expanded_box)

        # 绘制新的旋转矩形
        cv2.drawContours(line_image, [expanded_box], 0, (255, 0, 0), 2)
        
        x_min = np.min(expanded_box[:, 0])
        x_max = np.max(expanded_box[:, 0])
        y_min = np.min(expanded_box[:, 1])
        y_max = np.max(expanded_box[:, 1])

        # 确保边界框在图像范围内
        x_min = max(0, x_min)
        x_max = min(line_image.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(line_image.shape[0], y_max)

        # 裁剪 line_image
        image1 = image[y_min:y_max, x_min:x_max]
        height1, width1 = image1.shape[:2]
        long = width1
        short = height1
        print(f"长={long},宽={short}")
        # show(image1)
        cv2.imwrite('extracted_image.jpg', image1)
