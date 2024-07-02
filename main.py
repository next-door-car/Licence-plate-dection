import cv2
import numpy as np
from skimage.feature import hog
from model.eval_svm_cn import SVM_CN
from model.eval_svm_gray import SVM_GRAY
from model.train_svm_cn import SVM_CNT
from model.train_svm_gray import SVM_GRAYT
from model.lacations import Lacation
from model.segmentation import Segmentation


def predict_select(model_path,image_array,type, test_image_path):
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
    #预处理
    input_road = 'example/2.jpg'
    location = Lacation(input_road)
    image = location.picture_lacation()
    # cv2.imwrite('text.png',image)
    gass = Segmentation(image)
    image = gass.process_image()
    inffer = []
    if len(image) == 8: #第二张为.号
        image.pop(2)

    #现在 image[0]为汉字
    
    data_array = []
    svm_cn = predict_select(model_path = 'PT/svmCn.dat' ,image_array = image[0],type = 'cn',test_image_path=None)
    data_array.append(svm_cn)
    for i in range(1 , len(image)):
        svm_gray = predict_select(model_path = 'PT/svmGray.dat',image_array = image[i],type = 'gray',test_image_path=None)
        data_array.append(svm_gray)    
    print(f'Predicted class: {data_array}')


