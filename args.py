import cv2
import numpy as np
import argparse

cardtype = {"blue": "蓝色牌照",
            "green": "绿色牌照",
            "yellow": "黄色牌照"}
pic_size = {'open': 1,
            'blur': 3,
            'morphologyr': 4,
            'morphologyc': 19,
            'col_num_limit': 10,
            'row_num_limit': 21
            }
def args_parser():
    # 创建参数的解析对象
    parser = argparse.ArgumentParser(description='PyTorch garbage Training ')

    # 参数列表
    parser.add_argument('--Size', default=20, type=int, metavar='SZ', help='图片长宽')
    parser.add_argument('--MAX_WIDTH', default=1000, type=int, metavar='MAX', help='原始图片最大宽度')
    parser.add_argument('--Min_Area', default=2000, type=int, metavar='Min', help='车牌区域允许最大面积')
    parser.add_argument('--cardtype', default=cardtype, help='标签对应车牌类型')
    parser.add_argument('--Pic_size', default=pic_size, help='')

    # 解析参数
    args = parser.parse_args()
    return args
