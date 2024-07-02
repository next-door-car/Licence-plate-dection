import os
import shutil
from PIL import Image

def is_image(file_path):
    """
    检测文件是否为图像文件。
    """
    try:
        Image.open(file_path)
        return True
    except IOError:
        return False

def copy_images(folder_path):
    """
    检测子文件夹中的图像文件，并复制4份。
    """
    for root, dirs, files in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            image_files = [f for f in os.listdir(subdir_path) if is_image(os.path.join(subdir_path, f))]
            for image_file in image_files:
                image_path = os.path.join(subdir_path, image_file)
                for i in range(4):
                    new_image_path = os.path.join(subdir_path, f"{os.path.splitext(image_file)[0]}_copy{i+1}{os.path.splitext(image_file)[1]}")
                    shutil.copy2(image_path, new_image_path)

# 设置要处理的文件夹路径
folder_path = "train/annCh"

# 执行图像复制操作
copy_images(folder_path)
