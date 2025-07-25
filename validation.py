from ultralytics import YOLOv10
import os
os.environ[ 'CUDA_VISIBLE_DEVICES'] = '1'

model = YOLOv10('/disk16t/ycj/yolov10_research/runs/detect/ycj_train_v1/weights/best.pt')

model.val(data='/disk16t/ycj/yolov10_research/dataset/RVdataset.yaml', batch=16)