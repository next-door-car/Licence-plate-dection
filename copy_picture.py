import os
import shutil

# 指定文件夹路径
folder_path = 'train/annGray/T'

# 创建一个函数来复制图像文件
def copy_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 遍历所有文件
    for file_name in files:
        # 检查文件是否为图像文件（根据扩展名）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, file_name)
            
            # 创建4个副本
            for i in range(1, 5):
                # 构建副本文件名
                new_file_name = f"{os.path.splitext(file_name)[0]}_cpy{i}{os.path.splitext(file_name)[1]}"
                new_file_path = os.path.join(folder_path, new_file_name)
                
                # 复制文件
                shutil.copy(file_path, new_file_path)
                print(f"复制 {file_name} 为 {new_file_name}")

# 调用函数复制图像文件
copy_images(folder_path)
