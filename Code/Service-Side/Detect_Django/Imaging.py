from PIL import Image
import os

def resize_image(input_path, output_path, target_size_kb=100):
    # 打开图像文件
    img = Image.open(input_path)

    # 计算目标大小（字节）
    target_size_bytes = target_size_kb * 1024

    # 逐步减小质量以达到目标大小
    quality = 90
    while True:
        # 将图像保存到临时文件中
        img.save(output_path, optimize=True, quality=quality)

        # 获取临时文件大小
        file_size = os.path.getsize(output_path)

        # 如果文件大小小于目标大小，退出循环
        if file_size <= target_size_bytes:
            break

        # 减小质量
        quality -= 5

    # 关闭图像
    img.close()

# 调用函数并指定输入和输出路径
input_image_path = r"D:\Code\Detect_Django\img\databasePic\20240607_180606.jpg"
output_image_path = r"D:\Code\Detect_Django\img\databasePic\20240607_180606.jpg"
resize_image(input_image_path, output_image_path)
