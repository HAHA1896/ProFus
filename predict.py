# import cv2
# from ultralytics import YOLOv10

# def hsv2bgr(h, s, v):
#     h_i = int(h * 6)
#     f = h * 6 - h_i
#     p = v * (1 - s)
#     q = v * (1 - f * s)
#     t = v * (1 - (1 - f) * s)
    
#     r, g, b = 0, 0, 0

#     if h_i == 0:
#         r, g, b = v, t, p
#     elif h_i == 1:
#         r, g, b = q, v, p
#     elif h_i == 2:
#         r, g, b = p, v, t
#     elif h_i == 3:
#         r, g, b = p, q, v
#     elif h_i == 4:
#         r, g, b = t, p, v
#     elif h_i == 5:
#         r, g, b = v, p, q

#     return int(b * 255), int(g * 255), int(r * 255)

# def random_color(id):
#     h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
#     s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
#     return hsv2bgr(h_plane, s_plane, 1)

# if __name__ == "__main__":

#     model = YOLOv10("/disk16t/ycj/yolov10_research/runs/detect/train_v1074/weights/best.pt")

#     img = cv2.imread("/disk16t/ycj/yolov10_research/test.jpg")
#     results = model(img)[0]
#     names   = results.names
#     boxes   = results.boxes.data.tolist()

#     for obj in boxes:
#         left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
#         confidence = obj[4]
#         label = int(obj[5])
#         color = random_color(label)
#         cv2.rectangle(img, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
#         caption = f"{names[label]} {confidence:.2f}"
#         w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
#         cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
#         cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

#     cv2.imwrite("predict.jpg", img)
#     print("save done")    



from ultralytics import YOLO
import os
import cv2
import numpy as np

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 定义类别名称
class_name = ["fishing vessel", "speedboat", "engineering ship", "cargo ship", "yacht", "buoy", "cruise ship", "raft", "others"]

def predict(chosen_model, img, classes=[], conf=0.5):
    """使用模型进行预测"""
    print(f"输入图像尺寸：{img.shape}")
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

# 为每个类别生成随机颜色
num_classes = len(class_name)
colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(num_classes)]

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    """对图像进行预测并绘制检测框"""
    # 假设模型期望的输入尺寸为 640x640
    imgsz = 640
    img = cv2.resize(img, (imgsz, imgsz))
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            color = colors[int(box.cls[0]) % num_classes]
            score = box.conf[0].item()   # 预测得分
            label = f"{result.names[int(box.cls[0])]} {score:.2f}"  # 格式化标签
            # 绘制边界框
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
            # 添加标签
            cv2.putText(img, label,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, text_thickness)
    return img, results

def read_yolo_annotations(filename, img_size):
    """读取YOLO格式的标注文件"""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return []
    
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
    """在图像上绘制标注框"""
    img_cp = img.copy()  # 创建图像的副本以绘制边界框
    for class_id, xmin, ymin, xmax, ymax in annotations:
        color = colors[int(class_id) % num_classes]
        cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img_cp, class_names[class_id], (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    return img_cp

def main(image_path, model_path, conf_threshold=0.1):
    """主函数：对单张图片进行推理"""
    # 加载模型
    model = YOLO(model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    
    image4gt = image.copy()
    img_size = image.shape[:2]  # 获取图像的高度和宽度
    print(f"原始图像尺寸：{img_size}")
    
    # 模型推理并绘制预测框
    result_img, _ = predict_and_detect(model, image, classes=[], conf=conf_threshold)
    
    # 保存预测结果
    base_dir, filename = os.path.split(image_path)
    save_dir = os.path.join(base_dir, "prediction_results")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"pred_{filename}")
    cv2.imwrite(save_file, result_img)
    print(f"预测结果已保存至: {save_file}")
    
    # 读取并绘制标注框（如果存在）
    annotation_path = image_path.replace(".jpg", ".txt")
    annotations = read_yolo_annotations(annotation_path, img_size)
    
    if annotations:
        gt_img = draw_boxes(image4gt, annotations, class_name)
        save_gt_file = os.path.join(save_dir, f"gt_{filename}")
        cv2.imwrite(save_gt_file, gt_img)
        print(f"标注结果已保存至: {save_gt_file}")
    else:
        print("未找到标注文件")
    
    # 显示结果（可选）
    # cv2.imshow("Prediction", result_img)
    # if annotations:
    #     cv2.imshow("Ground Truth", gt_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # 用户需要修改以下参数
    IMAGE_PATH = "/disk16t/ycj/yolov10_research/dataset/test/test.jpg"  # 替换为实际图像路径
    MODEL_PATH = "/disk16t/ycj/yolov10_research/runs/detect/train_v1074/weights/best.pt"  # 模型路径
    CONF_THRESHOLD = 0.1  # 置信度阈值
    
    main(IMAGE_PATH, MODEL_PATH, CONF_THRESHOLD)