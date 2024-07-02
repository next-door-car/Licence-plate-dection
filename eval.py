import cv2
import numpy as np
from skimage.feature import hog
from model.eval_svm_cn import SVM_CN
from model.eval_svm_gray import SVM_GRAY
from model.train_svm_cn import SVM_CNT
from model.train_svm_gray import SVM_GRAYT

def predict_select(model_path,image_array, test_image_path, type):
    if type == 'cn':
        classifier = SVM_CN(model_path)  # 创建 SVM_CN 类的实例
    elif type == 'gray':
        classifier = SVM_GRAY(model_path)  # 创建 SVM_GRAY 类的实例
    else:
        raise ValueError("Invalid type provided")
    
    predicted_class = classifier.predict(image_array=image_array,image_path=test_image_path)
    return predicted_class

def train_select(type):
    if type == 'cn':
        svm_cn = SVM_CNT(C=2.5, gamma=0.05)
        svm_cn.train_svm_cn()

    elif type == 'gray':
        svm_gray = SVM_GRAYT(C=2.5, gamma=0.05)
        svm_gray.train_svm_gray()

if __name__ == '__main__':

    type='gray'
    # model_path="PT/svmCn.dat"
    # test_image_path="yc_picture/0.png"
    # # image_array ='1'
    # svm = predict_select(model_path = "PT/svmCn.dat",image_array =None,type = 'cn',test_image_path=test_image_path)
    # print(f'Predicted class: {svm}')
    train_select(type)


