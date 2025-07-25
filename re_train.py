#coding:utf-8
import os
os.environ[ 'CUDA_VISIBLE_DEVICES'] = '0'
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10s_test3.yaml"
#数据集配置文件
data_yaml_path = '/home/wushikai/mnt/disk_10t/WSK_SSD/project/yolov10-main/dataset/RVdataset.yaml'
#预训练模型
pre_model_name = '/home/wushikai/mnt/disk_10t/WSK_SSD/project/yolov10-main/checkpoints/yolov10s.pt'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = total_params * 4  # 每个参数占用4字节（32位浮点数）
    total_mb = total_bytes / (1024 ** 2)  # 转换为MB
    return total_params, total_mb

if __name__ == '__main__':
    #加载预训练模型
    model = YOLOv10(r"/home/wushikai/mnt/disk_10t/WSK_SSD/project/yolov10_research/runs/detect/train_v1064/weights/best.pt")

    with open('model_structure.txt', 'w') as f:
        f.write(str(model))
    num_p, mb_p = count_parameters(model)
    print(f"模型参数量为：{num_p}, 占内存：{mb_p}MB")
    #训练模型
    results = model.train(resume=True)
