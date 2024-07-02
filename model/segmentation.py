
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

if not os.path.exists('1'):
    os.makedirs('1')
if not os.path.exists('2'):
    os.makedirs('2')


class Segmentation:
    def __init__(self ,image):
        self.image = image
        
    
    def show(self, image):
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def angle(self, image):
        median_image = cv2.medianBlur(image, 5)
        median_image = 255 - median_image
        kerne3 = np.ones((30, 30), np.uint8)
        dilated_image = cv2.dilate(median_image, kerne3, iterations=1)

        edges = cv2.Canny(median_image, 70, 200, apertureSize=5, L2gradient=False)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        line_image = cv2.cvtColor(median_image, cv2.COLOR_GRAY2BGR)

        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        angles_array = np.array(angles)
        angles_array = angles_array[angles_array != 0.0]

        if len(angles_array) > 2:
            min_value = np.min(angles_array)
            max_value = np.max(angles_array)

            min_indices = np.where(angles_array == min_value)[0]
            max_indices = np.where(angles_array == max_value)[0]

            if len(min_indices) > 1:
                angles_array = np.delete(angles_array, min_indices[0])
            else:
                angles_array = angles_array[(angles_array != min_value)]

            if len(max_indices) > 1:
                angles_array = np.delete(angles_array, max_indices[0]) 
            else:
                angles_array = angles_array[(angles_array != max_value)]

        if len(angles_array) == 0:
            initial_average = 0
            filtered_angles_array = angles_array
        else:
            initial_average = np.mean(angles_array)
            filtered_angles_array = angles_array[np.abs(angles_array - initial_average) <= 10]

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

    def scaling(self, image):
        image_cope = image.copy()
        image1 = image.copy()
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 46])
        upper_blue = np.array([124, 255, 255])

        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        image[mask > 0] = [255, 255, 255]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(gray_image, 210, 255, cv2.THRESH_BINARY_INV)
        kerne2 = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kerne2) 

        average_angle = self.angle(closing)
        if average_angle != 0:
            (h, w) = closing.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, average_angle, 1.0)
            rotated_closing = cv2.warpAffine(closing, rotation_matrix, (w, h))

            if np.all(rotated_closing == 255):
                images = image_cope
            else:
                edges = cv2.Canny(rotated_closing, 100, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_points = np.vstack([contour for contour in contours])
                x, y, w, h = cv2.boundingRect(all_points)
                images = image1[y:y+h, x:x+w]
                if x == 0 and y == 0:
                    images = cv2.warpAffine(image1, rotation_matrix, (w, h))
        else:
            images = image_cope
        return images

    def two_value(self, img):
        ret, img_thre = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  
        closed = cv2.dilate(img_thre, kernel, iterations=1)
        return closed

    def process_image(self):
        image = self.image
        image1 = self.image
        images = self.scaling(image)

        gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        gray_image = self.two_value(gray_image)

        resized_image = cv2.resize(gray_image, (160, 40))
        resized_image = 255 - resized_image

        image2 = level(resized_image)
        image2 = cv2.GaussianBlur(image2, (3, 3), 0)

        pty = np.sum(image2 / 255, axis=0)

        average = sum(pty[:2]) / 2
        non_zero_indice = np.where(pty > 0)[0]
        non_zero_indices = np.where(pty > 2)[0]
        mean_values = []
        mean_values.append(0)
        values = []
        FLAG = 0

        if non_zero_indice[0] == 0:
            non_zero_indice = non_zero_indice[1:]  
        if non_zero_indices[0] == 0:
            non_zero_indices = non_zero_indices[1:]  
        if len(non_zero_indices) > 2:
            non_zero_indices = non_zero_indices[2:]

        print("All non-zero indices x:", non_zero_indices)

        if average == 0:
            FLAG = 1
            mean_values[0] = (non_zero_indice[0])
        else:
            for i in range(non_zero_indice[0], 9):
                if pty[i] == 0:
                    FLAG = 1
                    mean_values[0] = i
            if len(mean_values) == 0:
                for j in range(non_zero_indice[0], 9):
                    if pty[j] < 2:
                        FLAG = 1
                        mean_values[0] = j
            if len(mean_values) == 0:
                for t in range(non_zero_indice[0], 9):
                    if pty[t] < 3:
                        FLAG = 1
                        mean_values[0] = t

            if len(mean_values) == 0:
                print('picture one have a problem')
                sub_array = pty[:10]
                min_value = min(sub_array)
                FLAG = 1
                mean_values[0] = sub_array.index(min_value)

        values = [mean_values[0]]

        print("Mean values for x indices with gap > 3:")
        for i in range(len(non_zero_indices) - 1):
            if non_zero_indices[i + 1] - non_zero_indices[i] >= 2:
                mean_value = int((non_zero_indices[i] + non_zero_indices[i + 1]) / 2)
                if FLAG == 1:
                    FLAG = 0
                else:
                    mean_values.append(mean_value)
                values.append(non_zero_indices[i+1])

        if len(mean_values) > 2 and len(values) > 2:
            if mean_values[1] - mean_values[0] >= 35:
                if mean_values[1] - values[1] > 10:
                    mean_values.insert(1, values[1])

        if len(mean_values) < 8:
            mean_values.append(158)

        if mean_values[0] < 10 and mean_values[1] < 10:
            mean_values.pop(0)
        print(f'mean={mean_values}')
        cropped_images = []
        for i in range(len(mean_values) - 1):
            x1 = mean_values[i]
            x2 = mean_values[i + 1]
            images_all = image2[:, x1:x2]
            #cv2.imwrite(f'1/{i}.png', images_all)
            cropped_images.append(images_all)  # 将裁剪后的图像添加到列表中

        # print("Stored mean values:", mean_values)

        # plt.figure(figsize=(10, 5))
        # plt.bar(np.arange(len(pty)), pty, color='black')
        # plt.title('Vertical Projection Histogram')
        # plt.xlabel('Column Index')
        # plt.ylabel('Number of White Pixels')
        # plt.show()
        return cropped_images