import cv2
import os
from tqdm import tqdm

def convertImg(dirPath):
    try:
        img = cv2.imread(dirPath)  # 读图像数据
        new_path = dirPath.replace("project/RVSSD/data/RVDataset","RVDataset_recover").replace(".png",".jpg")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, img)  # 另存同名，同扩展名图像。不破坏数据集
        print("写入新图片：", new_path)
    except cv2.error as e:
        print("图片损坏,图片名称：", fileName, e)
    print("全部图像已另存！")

def check_image(image_path):
    # print("check:", image_path)
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        print(f"捕捉到异常: {e} - 图像: {image_path}")
        return False

    return True

file_path = '/home/wushikai/mnt/disk_10t/WSK_SSD/RVDataset/data_all.txt'

with open(file_path, "r") as file:
    total_lines = sum(1 for line in file)

with open(file_path, 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in file: #tqdm(file, total=total_lines, desc="Processing"):
        image_file = line.strip()+'.png'
        convertImg(image_file.replace("camera","rvm2"))
    print("check done!")
