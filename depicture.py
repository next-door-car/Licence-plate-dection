import os
import glob

def delete_images_in_subfolders(folder_path):
    # 获取所有子文件夹路径
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    for subfolder in subfolders:
        # 获取子文件夹下所有图片文件
        images = glob.glob(os.path.join(subfolder, '*.*'))
        for image in images:
            try:
                os.remove(image)
                print(f"Deleted: {image}")
            except Exception as e:
                print(f"Error deleting {image}: {e}")

# 指定主文件夹路径
main_folder = 'train/annGray'

# 删除所有子文件夹中的图片
delete_images_in_subfolders(main_folder)
