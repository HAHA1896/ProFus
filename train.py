#coding:utf-8
import os
os.environ[ 'CUDA_VISIBLE_DEVICES'] = '1'
from ultralytics import YOLOv10
# 模型配置文件
model_yaml_path = "/disk16t/ycj/yolov10_research/ultralytics/cfg/models/v10/yolov10s_test4.yaml"
# 数据集配置文件
data_yaml_path = '/disk16t/ycj/yolov10_research/dataset/RVdataset.yaml'
# 预训练模型
pre_model_name = '/disk16t/ycj/yolov10_research/checkpoints/yolov10s.pt'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = total_params * 4  # 每个参数占用4字节（32位浮点数）
    total_mb = total_bytes / (1024 ** 2)  # 转换为MB
    return total_params, total_mb

if __name__ == '__main__':
    #加载预训练模型
    model = YOLOv10(model_yaml_path).load(pre_model_name)

    with open('model_structure_4.txt', 'w') as f:
        f.write(str(model))
    num_p, mb_p = count_parameters(model)
    print(f"模型参数量为：{num_p}, 占内存：{mb_p}MB")
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=10,
                          batch=2,
                          name='ycj_train_v1',
                          weight_decay=0.0015)




# from ultralytics import YOLOv10

# model = YOLOv10()
# # If you want to finetune the model with pretrained weights, you could load the 
# # pretrained weights like below
# # model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# # or
# # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# # model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

# model.train(data='/disk16t/ycj/env_monitor_dataset/env_monitor_dataset/env_monitor_dataset.yaml', epochs=500, batch=256, imgsz=640)















