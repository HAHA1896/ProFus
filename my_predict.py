'''
Author: Shikai Wu weskieykai@163.com
Date: 2024-09-12 10:44:44
LastEditors:  
LastEditTime: 2024-09-12 10:45:13
FilePath: \project\yolov10_research\pridict.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
from ultralytics import YOLOv10
import os
import cv2
import numpy as np
os.environ[ 'CUDA_VISIBLE_DEVICES'] = '0'

class_name = ["fishing vessel", "speedboat", "engineering ship", "cargo ship", "yacht", "buoy", "cruise ship", "raft", "others"]

# def predict(chosen_model, img, classes=[], conf=0.5):
#     if classes:
#         results = chosen_model.predict(img, classes=classes, conf=conf)
#     else:
#         results = chosen_model.predict(img, conf=conf)

#     return results

# num_classes = len(class_name)
# colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(num_classes)]

# def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
#     results = predict(chosen_model, img, classes, conf=conf)
#     for result in results:
#         for box in result.boxes:
#             color = colors[int(box.cls[0]) % num_classes]
#             score = box.conf[0].item()   # 假设box对象有一个conf属性，表示预测得分
#             label = f"{result.names[int(box.cls[0])]} {score:.2f}"  # 格式化得分，保留两位小数
#             cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
#                           (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
#             cv2.putText(img, label,
#                         (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
#                         cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
#     return img, results

def predict(chosen_model, imgs, classes=[], conf=0.5):
    if classes:
        print(f"这个是predict里的输入图像数量={len(imgs)}")
        results = chosen_model.predict(imgs, classes=classes, conf=conf)
        
    else:
        print(f"这个是predict,else里的输入图像数量={len(imgs)}")
        results = chosen_model.predict(imgs, conf=conf)
    return results

num_classes = len(class_name)
colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(num_classes)]

def predict_and_detect(chosen_model, imgs, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    print(f"这个是predict_and_detect里的输入图像数量={len(imgs)}")
    results = predict(chosen_model, imgs, classes, conf=conf)
    result_images = []
    for i, img in enumerate(imgs):
        img_copy = img.copy()  # 创建图像副本，避免修改原始图像
        for result in results:
            for box in result.boxes:
                color = colors[int(box.cls[0]) % num_classes]
                score = box.conf[0].item()   # 假设box对象有一个conf属性，表示预测得分
                label = f"{result.names[int(box.cls[0])]} {score:.2f}"  # 格式化得分，保留两位小数
                cv2.rectangle(img_copy, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
                cv2.putText(img_copy, label,
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
        result_images.append(img_copy)
    return result_images, results

def read_yolo_annotations(filename, img_size):
    with open(filename, 'r') as file:
        lines = file.readlines()
    annotations = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        # 将中心点坐标和宽高转换为左上角和右下角坐标
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2

        # 将比例坐标转换为像素坐标
        xmin = int(xmin * img_size[1])
        xmax = int(xmax * img_size[1])
        ymin = int(ymin * img_size[0])
        ymax = int(ymax * img_size[0])

        annotations.append((class_id, xmin, ymin, xmax, ymax))

    return annotations

def draw_boxes(img, annotations, class_names):
    img_cp = img.copy()  # 创建图像的副本以绘制边界框
    for class_id, xmin, ymin, xmax, ymax in annotations:
        color = color = colors[int(class_id) % num_classes]
        cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img_cp, class_names[class_id], (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    return img_cp

model = YOLOv10('/disk16t/ycj/yolov10_research/runs/detect/ycj_train_v1/weights/best.pt')

# model.val(data='/home/wushikai/mnt/disk_10t/WSK_SSD/project/yolov10-main/dataset/RVdataset.yaml', batch=1)
# read the image
file_path = "/disk16t/ycj/yolov10_research/dataset/test/test_yolo.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in file:
        image_file = line.strip()
        print(f"这个是image——file:{image_file}")
        radar_file = image_file.replace("camera","radar2")
        print(f"这个是radar——file:{radar_file}")
        rvm_file =image_file.replace("camera",'rvm2')
        print(f"这个是rvm——file:{rvm_file}")
        image = cv2.imread(image_file)
        print(f"这个是image的shape={image.shape}")
        radar_image = cv2.imread(radar_file)
        print(f"这个是radar_image的shape={radar_image.shape}")

        rvm_image = cv2.imread(rvm_file)
        print(f"这个是rvm_image的shape={rvm_image.shape}")
        
        images = [image, radar_image, rvm_image]
        image4gt = image.copy()
        img_size = image.shape[:2]  # 获取图像的高度和宽度

        result_img, _ = predict_and_detect(model, images, classes=[], conf=0.1)
        # save_file = image_file.replace("RVDataset_recover","RVDataset_test")
        save_file = image_file.replace("test","detect")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        # cv2.imshow("Image", result_img)
        cv2.imwrite(save_file, result_img)
        print("predict save in:", save_file)

        annotations = read_yolo_annotations(image_file.replace("test","labels").replace(".jpg",".txt"), img_size)
        gt_img = draw_boxes(image4gt, annotations, class_name)
        save_gt_file = save_file.replace(".jpg","_gt.jpg")
        cv2.imwrite(save_gt_file, gt_img)
        print("ground true save in:", save_gt_file)
        # cv2.waitKey(0)
