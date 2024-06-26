import cv2
from matplotlib import pyplot as plt

img_ = cv2.imread('extracted_image.jpg')  # 读取图片
img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image = cv2.dilate(img_gray, kernel)
# 查找轮廓
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
words = []
word_images = []
for item in contours:
    word = []
    rect = cv2.boundingRect(item)
    x = rect[0]
    y = rect[1]
    weight = rect[2]
    height = rect[3]
    word.append(x)
    word.append(y)
    word.append(weight)
    word.append(height)
    words.append(word)
words = sorted(words,key=lambda s:s[0],reverse=False)
i = 0
for word in words:
    # 根据轮廓的外接矩形筛选轮廓
    if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
        i = i+1
        splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        splite_image = cv2.resize(splite_image,(25,40))
        word_images.append(splite_image)

for i,j in enumerate(word_images):
    plt.subplot(1,7,i+1)
    plt.imshow(word_images[i],cmap='gray')
plt.show()
