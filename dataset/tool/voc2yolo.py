'''
Author: Shikai Wu weskieykai@163.com
Date: 2024-08-20 16:05:45
LastEditors:  
LastEditTime: 2024-08-20 16:34:43
FilePath: \RVDataset\voc2yolo.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

#coding:utf-8
from __future__ import print_function
 
import os
import random
import glob
import xml.etree.ElementTree as ET
 
def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return width, height, objects
 
 
def voc2yolo(filename):
    classes_dict = {}
    with open("classes.names") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()
            classes_dict[class_name] = idx
    
    width, height, objects = xml_reader(filename)
 
    lines = []
    for obj in objects:
        x, y, x2, y2 = obj['bbox']
        class_name = obj['name']
        label = classes_dict[class_name]
        cx = (x2+x)*0.5 / width
        cy = (y2+y)*0.5 / height
        w = (x2-x)*1. / width
        h = (y2-y)*1. / height
        line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
        lines.append(line)
 
    txt_name = filename.replace(".xml", ".txt")
    with open(txt_name, "w") as f:
        f.writelines(lines)
        print('save to:', txt_name)
 

 
if __name__ == "__main__":
# 打开文本文件
    with open('/home/wushikai/mnt/disk_10t/WSK_SSD/RVDataset/data_all.txt', 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 打印当前行
            xml_path = line.strip()+'.xml'
            print(xml_path)  # 使用 strip() 去除行末尾的换行符
            voc2yolo(xml_path)

 
    # imglist = get_image_list("images")
    # imglist2file(imglist)