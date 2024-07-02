# import os

# folder_path = r'success'  # 文件夹路径
# file_extension = '.jpg'  # 文件扩展名

# count = 1

# for filename in os.listdir(folder_path):
#     if filename.endswith(file_extension):
#         current_name = os.path.join(folder_path, filename)
#         new_name = os.path.join(folder_path, str(count) + file_extension)
#         os.rename(current_name, new_name)
#         count += 1
#         print(f"Renamed {current_name} to {new_name}")


# import os
# folder_path = r'success'  # 文件夹路径
# file_extension = '.jpg'  # 文件扩展名

# count = 0

# for filename in os.listdir(folder_path):
#     if filename.endswith(file_extension):
#         current_name = os.path.join(folder_path, filename)
#         new_name = os.path.join(folder_path, f"{count:02d}{file_extension}")
        
#         # 如果目标文件名已经存在，则跳过
#         if os.path.exists(new_name):
#             print(f"File {new_name} already exists. Skipping...")
#             continue
        
#         os.rename(current_name, new_name)
#         count += 1
