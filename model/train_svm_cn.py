import cv2
import os
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


index = ["鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", 
         "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼",
           "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙","川"]


def preprocess_hog(images):
    processed_images = []
    for image in images:
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       block_norm='L2-Hys', transform_sqrt=True)
        processed_images.append(features)
    return processed_images

def deskew(image):
    moments = cv2.moments(image)
    if abs(moments['mu02']) < 1e-2:
        return image.copy()
    skew = moments['mu11'] / moments['mu02']
    M = np.float32([[1, skew, -0.5 * image.shape[0] * skew], [0, 1, 0]])
    deskewed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return deskewed

def augment_image(image):
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

class SVM_CNT:
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.count = 0

    def train(self, samples, responses):
        samples = samples.astype(np.float32)
        responses = responses.astype(np.int32)
        train_data = cv2.ml.TrainData_create(samples, cv2.ml.ROW_SAMPLE, responses)
        self.model.train(train_data)

    def train_svm_cn(self):
        save_path = "PT/svmCn.dat"
        chars_train = []
        chars_label = []
        root_folder = "train/annCh"
        folders = sorted([folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))])
        total_files = sum([len(files) for r, d, files in os.walk(root_folder)])

        with tqdm(total=total_files, desc="Processing images", ascii=True) as pbar:
                for folder in folders:
                    folder_path = os.path.join(root_folder, folder)
                    #if folder in index:
                    label = index[self.count]
                    self.count =self.count+ 1
                    for root, dirs, files in os.walk(folder_path):
                        for filename in files:
                            filepath = os.path.join(root, filename)
                            digit_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                            digit_img = cv2.resize(digit_img, (16, 16))
                            augmented_images = augment_image(digit_img)
                            for img in augmented_images:
                                chars_train.append(deskew(img))
                                chars_label.append(self.count - 1)  #使用对应的index序列映射值到chars_label
                            pbar.update(1)

        chars_train = preprocess_hog(chars_train)
        chars_train = np.array(chars_train).astype(np.float32)
        chars_label = np.array(chars_label).astype(np.int32)
        chars_train = chars_train.reshape(-1, chars_train.shape[1])

        with tqdm(total=1, desc="Training SVM", ascii=True) as pbar:
                self.train(chars_train, chars_label)
                self.model.save(save_path)
                pbar.update(1)

 


    