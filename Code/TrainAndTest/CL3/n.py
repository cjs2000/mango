import os


def rename_jpg_files(directory):
    # 获取目录中的所有文件列表
    files = os.listdir(directory)

    # 过滤出所有的jpg文件
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]

    # 对jpg文件进行排序
    jpg_files.sort()

    # 重命名每个jpg文件
    for i, filename in enumerate(jpg_files, start=1):
        new_name = f"l_{i}.jpg"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} to {new_path}')

# 使用方法
directory_path = 'data/Mango_split/val/Tainong Mango'  # 替换为你的文件夹路径
rename_jpg_files(directory_path)
