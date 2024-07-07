import json
import os
import cv2
from ultralytics import YOLO

from django.db.backends import mysql
from django.shortcuts import render
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import HttpResponse, FileResponse
from rembg import remove
import ctypes
from ctypes import *
import shutil


from PIL import Image
import numpy as np
import pymysql
from mysql.connector import Error
import base64
import time
from datetime import datetime
from PIL import Image
from torchutils import get_torch_transforms
from model.shufflenet_ECA_GC import *

os.add_dll_directory(r"D:\ER\opencv\build\include")
lib = CDLL(r"D:\Code\Germinate_dll\Germinate_dll\Germinate_dll\dll\Germinatedll.dll",winmode=0)# winmode=0

def make_square(image):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 找到较长的一边
    if height > width:
        # 计算要补充的宽度
        padding = (height - width) // 2
        # 使用cv2.copyMakeBorder函数补充边界，top, bottom, left, right依次为补充的像素数
        square_image = cv2.copyMakeBorder(image, 0, 0, padding, height - width - padding, cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])
    else:
        # 计算要补充的高度
        padding = (width - height) // 2
        # 使用cv2.copyMakeBorder函数补充边界，top, bottom, left, right依次为补充的像素数
        square_image = cv2.copyMakeBorder(image, padding, width - height - padding, 0, 0, cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])

    return square_image




def yoloProcess():
    model = YOLO(r'N/best.pt')  # load a custom model
    # Predict with the mode
    input_path = r"D:\Code\Detect_Django\img\s.jpg"
    results = model.predict(source=input_path)
    image = cv2.imread(input_path)

    result = results[0]
    size = result.boxes.shape[0]
    for i in range(size):
        box = result.boxes[i]

        # 将类ID转换为字符串并获取第九个字符（如果存在）
        id = str(box.cls)
        ninth_char = id[8] if len(id) > 8 else None

        # 获取置信度
        conf = box.conf.tolist()[0]

        # 根据置信度和类ID筛选
        if conf > 0.1 and ninth_char == '0':
            # 获取边界框坐标
            print(conf)
            cords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, cords)
            cropped_image = image[y1:y2, x1:x2]
            square_image = make_square(cropped_image)
            cv2.imwrite("N/b.jpg", square_image)



def predict_mango():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    classes_names = ['Ao Mango', 'GuiQi Mango', 'Jinhuang Mango', 'Tainong Mango']
    #image_path = r"D:\Code\Detect_Django\img\s.jpg"
    image_path = "N/b.jpg"
    weights_path = r"D:\Code\Detect_Django\model\best.pth"

    yoloProcess()

    data_transforms = get_torch_transforms(img_size=224)
    # train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']

    model = shufflenet_eca_gc()  # todo 模型选择
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)


    # 读取图片
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)

    label_id = torch.argmax(output).item()
    # 获取预测类别的置信度
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence = str(probabilities[0, label_id].item())

    print("置信度：" + confidence)

    predict_name = classes_names[label_id]

    id = str(label_id)
    predict_name = classes_names[label_id]
    print(f"{image_path}'s result is {predict_name}")
    return id,confidence

def query_database(result_index):
    # 创建数据库连接
    connection = pymysql.connect(host="localhost", port=3306, user="root", password="123456", database="detect")
    if connection.open:
        print("连接成功")
    else:
        print("连接失败")
    # 使用 `SELECT` 语句查询数据库
    cursor = connection.cursor()
    sql = "SELECT mangoID,mangoName,mangoColor, mangoInfo FROM mango WHERE mangoID = %d" % result_index
    cursor.execute(sql)
    result = cursor.fetchone()

    # 将查询结果转换为 JSON 格式
    response_data = {}
    if result is not None:
        response_data['mangoID'] = result[0]
        response_data['mangoName'] = result[1]
        response_data['mangoColor'] = result[2]
        response_data['mangoInfo'] = result[3]

    # 关闭数据库连接
    connection.close()
    return response_data



class ImageProcessor:
    def __init__(self, image_data):
        self.image_data = image_data

    def process_image(self):
        try:
            output = remove(self.image_data)
            return output
        except Exception as e:
            return None




def insert_data(host, user, password, database, table, sum_value, gerNum_value, gerRate_value, runTime_value):
    # 连接到数据库
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    # 创建一个游标对象
    cursor = connection.cursor()

    # 获取当前时间戳
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 数据
    data = (current_time, sum_value, gerNum_value, gerRate_value, runTime_value)

    # 准备插入数据的 SQL 查询
    insert_query = f"INSERT INTO {table} (time, sum, gerNum, gerRate, runTime) VALUES (%s, %s, %s, %s, %s)"

    # 执行插入操作
    cursor.execute(insert_query, data)

    # 提交更改到数据库
    connection.commit()

    # 关闭游标和连接
    cursor.close()
    connection.close()


class Germinate(APIView):
    # 指定解析器类，用于处理多部分表单数据
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        start_time = time.time()
        # 检查请求数据中是否包含名为'image'的文件部分
        if 'image' not in request.data:
            return Response({'error': 'No file part'}, status=status.HTTP_400_BAD_REQUEST)

        # 获取图像文件对象
        image = request.data['image']

        # 将图像的二进制数据保存到文件
        with open(r'D:\Code\Germinate_dll\Germinate_dll\Germinate_dll\image\Original_image_Germinate.jpg', 'wb') as output_image:
            output_image.write(image.read())

        # 声明函数返回值类型
        lib.Process.restype = ctypes.c_char_p
        # 记录开始时间

        result = lib.Process()

        # 将C字符串转换为Python字符串
        result_str = result.decode('utf-8')
        print(result_str)
        # 使用 "#" 字符分割字符串
        parts = result_str.split("#")
        # 提取两个数字并将它们转换为整数
        if len(parts) == 2:
            gerNum = float(parts[0])
            sum = float(parts[1])
            print("发芽的数量:", gerNum)
            print("种子总数:", sum)
        else:
            print("无法提取两个数字")
        germinate_rate = gerNum / sum
        # 使用 round 函数来将结果保留后四位小数
        germinate_rate = round(germinate_rate, 4)
        germinate_rate_str=str(germinate_rate)
        print("发芽率:" + germinate_rate_str)
        # 将图像数据编码为Base64字符串
        with open(r"D:\Code\Germinate_dll\Germinate_dll\Germinate_dll\result\connect_seed.jpg", 'rb') as image_file_1:
            image_data_base64_connect_seed = base64.b64encode(image_file_1.read()).decode('utf-8')
        with open(r"D:\Code\Germinate_dll\Germinate_dll\Germinate_dll\result\connect_sprout.jpg", 'rb') as image_file_2:
            image_data_base64_connect_sprout = base64.b64encode(image_file_2.read()).decode('utf-8')

        end_time = time.time()
        # 计算运行时间
        runtime = round(end_time - start_time, 4)
        insert_data("localhost", "root", "123456", "germination", "infor", sum, gerNum,  germinate_rate_str, runtime)
        # 生成 JSON 数据
        data = {
            'image1': image_data_base64_connect_seed,
            'image2': image_data_base64_connect_sprout,
            'sum': sum,
            'gerRate': germinate_rate_str,
            'gerNum': gerNum,
            'runTime': runtime
        }
        return Response(data, status = status.HTTP_200_OK)



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


def insert_mango_data(host, user, password, database, table,runTime,confidence,id):
    # 连接到数据库
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    # 创建一个游标对象
    cursor = connection.cursor()

    # 获取当前时间戳
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    source_file = r"D:\Code\Detect_Django\img\s.jpg"

    # 目标目录
    target_directory = r"D:\Code\Detect_Django\img\databasePic"

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    currenttime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构造目标文件路径
    target_file = os.path.join(target_directory, f"{currenttime}.jpg")

    # 复制并重命名文件
    shutil.copy2(source_file, target_file)

    picPath = str(target_file)

    #resize_image(picPath, picPath)

    mangoName = id

    # 数据
    data = (current_time, picPath,runTime,confidence,mangoName)

    # 准备插入数据的 SQL 查询
    insert_query = f"INSERT INTO {table} (time, picPath,runTime,confidence,mangoName) VALUES (%s, %s, %s, %s, %s)"

    # 执行插入操作
    cursor.execute(insert_query, data)

    # 提交更改到数据库
    connection.commit()

    # 关闭游标和连接
    cursor.close()
    connection.close()




class Mango(APIView):
    # 指定解析器类，用于处理多部分表单数据
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request, format=None):
        start_time = time.time()
        if 'image' not in request.data:
            return Response({'error': 'No file part'}, status=status.HTTP_400_BAD_REQUEST)

        # 获取图像文件对象
        image = request.data['image']
        # 将图像的二进制数据保存到文件
        with open(r'img\s.jpg',
                  'wb') as output_image:
            output_image.write(image.read())

        id,confidence = predict_mango()

        end_time = time.time()
        runTime = str(round(end_time - start_time, 2))
        # 生成 JSON 数据
        data = {
            'id': id,
            'confidence': confidence,
            'runTime': runTime
        }


        insert_mango_data("localhost", "root", "123456", "mango", "infor",runTime,confidence,id)

        return Response(data, status=status.HTTP_200_OK)






def get_table_data_as_json(host, user, password, database, table_name):
    try:
        # 连接到数据库
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # 创建一个游标
        cursor = connection.cursor()

        # 执行查询语句，获取所有项
        query = f"SELECT * FROM {table_name} ORDER BY time ASC"
        cursor.execute(query)

        # 获取查询结果
        result = cursor.fetchall()
        # 获取查询结果的大小
        result_size = len(result)
        # 关闭游标和数据库连接
        cursor.close()
        connection.close()

        # 转换结果为JSON格式

        data = {
            "result_size": result_size,
            "items": []
        }

        for row in result:
            picPath = row[1]
            with open(picPath,
                      'rb') as image_file_1:
                image_data_base64 = base64.b64encode(image_file_1.read()).decode('utf-8')
            row_data = {
                'time': row[0].strftime("%Y-%m-%d %H:%M:%S"),
                'runTime': row[2],
                'confidence': row[3],
                'image': image_data_base64,
                'id': row[4]
            }
            data["items"].append(row_data)

        # 将数据转为JSON格式
       # json_data = json.dumps(data, indent=4)
        return data
    except Exception as e:
        return f"Error: {str(e)}"


def Mod_time_database(host, user, password, database, table_name, t):
    connection = None  # 初始化 connection 变量
    try:
        # 连接到数据库
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        cursor = connection.cursor()
        # 查询日期最新的那条记录
        select_query = f"SELECT time FROM {table_name} ORDER BY time DESC LIMIT 1"
        cursor.execute(select_query)
        record = cursor.fetchone()

        if record:
            # 获取最新记录的时间值
            latest_time = record[0]

            # 更新该记录的 runTime 值
            update_query = f"UPDATE {table_name} SET runTime = %s WHERE time = %s"
            cursor.execute(update_query, (t, latest_time))
            connection.commit()
            print("Record updated successfully")

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("MySQL connection is closed")



class GetInfo(APIView):
    def get(self, request):
        # 在这里编写处理GET请求的逻辑
        json_data = get_table_data_as_json('localhost', 'root', '123456', 'mango', 'infor')
        return Response(json_data, status=status.HTTP_200_OK)




class RecordTime(APIView):
    def get(self, request):
        time_value = request.query_params.get('time', None)

        if time_value is not None:
            # 处理time_value，比如记录到数据库或者其他操作
            # 假设我们只是返回这个值
            data = {'time': time_value}
            Mod_time_database('localhost', 'root', '123456', 'mango', 'infor',time_value)
            print(data)
            return Response(data, status=status.HTTP_200_OK)
        else:
            # 如果没有'time'参数，返回错误信息
            return Response({'error': 'Time parameter is required'}, status=status.HTTP_400_BAD_REQUEST)


def delete_data_by_time_and_item_id(item_id):
    try:
        # 建立数据库连接
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="123456",
            db="mango"
        )

        # 创建游标
        cursor = connection.cursor()

        # 查询数据并按时间排序
        query = "SELECT * FROM infor ORDER BY time ASC"
        cursor.execute(query)

        # 获取查询结果
        rows = cursor.fetchall()

        if item_id < len(rows):
            # 获取要删除的数据的详细信息
            data_to_delete = rows[item_id]
            time_to_delete = data_to_delete[0]  # 假设 time 是第一列
            picPath_to_delete = data_to_delete[1]  # 假设 picPath 是第三列

            # 删除图片文件
            if os.path.exists(picPath_to_delete):
                os.remove(picPath_to_delete)
                print(f"成功删除图片文件: {picPath_to_delete}")
            else:
                print(f"图片文件不存在: {picPath_to_delete}")

            # 构建删除数据的SQL语句
            delete_query = "DELETE FROM infor WHERE time = %s"
            cursor.execute(delete_query, (time_to_delete,))

            # 提交事务
            connection.commit()

            # 打印删除的 picPath
            print(f"成功删除第 {item_id} 项数据，picPath: {picPath_to_delete}")

        else:
            print(f"数据项 {item_id} 不存在")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cursor.close()
        connection.close()


class DeleteItemView(APIView):
    def get(self, request):
        item_id = request.GET.get('id')  # 通过request.GET获取id参数
        item_id = int(item_id)
        delete_data_by_time_and_item_id(item_id)
        json_data = get_table_data_as_json('localhost', 'root', '123456', 'mango', 'infor')
        return Response(json_data, status=status.HTTP_200_OK)


def clear_table(host, port, user, password, database, table_name):
    # 连接数据库
    conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

    # 使用 truncate 语句清空表
    cursor = conn.cursor()
    cursor.execute("truncate table {}".format(table_name))
    conn.commit()
    cursor.close()

    # 关闭数据库连接
    conn.close()


def delete_all_images_in_folder(folder_path):
    try:
        # 检查文件夹是否存在
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 遍历文件夹中的所有文件
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 检查文件是否是图片文件
                    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        os.remove(file_path)
                        print(f"成功删除图片文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 失败: {e}")
        else:
            print(f"文件夹不存在: {folder_path}")
    except Exception as e:
        print(f"发生错误: {e}")


class ClearItemView(APIView):
    def get(self, request):
        # 在这里编写处理GET请求的逻辑
        clear_table("localhost", 3306, "root", "123456", "mango", "infor")
        json_data = {"message": "数据表已清空", "status": "成功"}
        folder_path = r"D:\Code\Detect_Django\img\databasePic"
        delete_all_images_in_folder(folder_path)
        return Response(json_data, status=status.HTTP_200_OK)







