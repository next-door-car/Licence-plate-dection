import cv2
import numpy as np
from skimage.feature import hog


class SVM_GRAY:
    def __init__(self, model_path):
        self.model = cv2.ml.SVM_load(model_path)

    def preprocess_image(self, image_path):
        # 读取图像并转换为灰度图
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 调整图像大小为模型训练时的大小（16x16）
        resized = cv2.resize(gray, (16, 16))
        return resized

    def deskew(self, image):
        # 计算图像的倾斜角度
        moments = cv2.moments(image)
        skew = moments['mu11'] / moments['mu02']
        # 构建旋转矩阵并对图像进行旋转
        M = np.float32([[1, skew, -0.5 * image.shape[0] * skew], [0, 1, 0]])
        deskewed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return deskewed

    def preprocess_hog(self, image):
        # 对图像应用 HOG 特征提取
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        return features

    def predict(self,image_array=None, image_path=None):
        # 读取并预处理测试图像
        # test_image = self.preprocess_image(image_path)
        #image = cv2.imread(image_path)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 调整图像大小为模型训练时的大小（16x16）
        #resized = cv2.resize(gray, (16, 16))
        if image_path is not None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (16, 16))
        elif image_array is not None:
            image = cv2.resize(image_array, (16, 16))
        else:
            raise ValueError("Either image_array or image_path must be provided")
        deskewed_image = self.deskew(image)
        hog_features = self.preprocess_hog(deskewed_image)

        # 将特征转换为正确的形状和类型
        test_features = np.array(hog_features).reshape(1, -1).astype(np.float32)

        # 使用模型进行预测
        result = self.model.predict(test_features)[1]

        # 返回预测结果
        predicted_char = chr(int(result[0][0]))
        return predicted_char