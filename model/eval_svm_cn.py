# svm_cn.py

import cv2
import numpy as np
from skimage.feature import hog

class SVM_CN:
    def __init__(self, model_path):
        self.model = cv2.ml.SVM_load(model_path)
        if self.model is None:
            print(f"Failed to load SVM model from {model_path}")

    def predict(self, image_array=None, image_path=None):
        if image_path is not None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (16, 16))
        elif image_array is not None:
            image = cv2.resize(image_array, (16, 16))
        else:
            raise ValueError("Either image_array or image_path must be provided")
        image = self.deskew(image)
        features = self.preprocess_hog(image)
        features = np.array(features).astype(np.float32)
        predicted_label = int(self.model.predict(features.reshape(1, -1))[1].ravel()[0])

        index = ["鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", 
                 "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼",
                 "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙", "川"]

        if 0 <= predicted_label < len(index):
            predicted_class = index[predicted_label]
        else:
            predicted_class = "Unknown"

        return predicted_class

    def preprocess_hog(self, image):
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', transform_sqrt=True)
        return features

    def deskew(self, image):
        moments = cv2.moments(image)
        if abs(moments['mu02']) < 1e-2:
            return image.copy()
        skew = moments['mu11'] / moments['mu02']
        M = np.float32([[1, skew, -0.5 * image.shape[0] * skew], [0, 1, 0]])
        deskewed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return deskewed

    def augment_image(self, image):
        rows, cols = image.shape
        augmented_images = [image]

        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows))
            augmented_images.append(rotated)

        for dx in [-2, 2]:
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            translated = cv2.warpAffine(image, M, (cols, rows))
            augmented_images.append(translated)

        return augmented_images
