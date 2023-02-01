import cv2
import os
import numpy as np
import json
import mmcv
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from glob import glob
import ast
from mmrotate.core import poly2obb_np

def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a

def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def draw_rbboxes(ax, bboxes, color='g', alpha=0.3, thickness=2):
    """Draw oriented bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 5).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        poly = np.int0(np.array([p1, p2, p3, p4]))
        polygons.append(Polygon(poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax

# arirang json to txt

ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
# out_path = '/data/2_data_server/cv_data/arirang/val/annfiles/'

# ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
label_test = []

for i in glob(ann_path+'*.json'):
    # print(i)
    with open(i) as f:
        json_data = json.load(f)
    img_id = json_data['features'][0]['properties']['image_id']
    for j in range(len(json_data['features'])):

        bbox_info = json_data['features'][j]['properties']['object_imcoords']
        bbox_info = ast.literal_eval(bbox_info)
        bbox_info = list(bbox_info)
        
        bbox_label = json_data['features'][j]['properties']['type_name'].replace(" ","-")
        bbox_id = json_data['features'][j]['properties']['type_id']

        if bbox_label == "military-aircraft":
            print(img_id)
            exit()
        if label_test == []:
            # label_test.append(bbox_id)
            label_test.append(bbox_label)
        if bbox_label not in label_test:
            # label_test.append(bbox_id)
            label_test.append(bbox_label)

       
        # first [:4] 지운 후
        # if j == 0:
        #     with open(out_path+img_id[:-4]+'.txt',"w") as (fw):
        #         for k in range(len(bbox_info)):
        #             fw.write(str(int(bbox_info[k])))
        #             fw.write(" ")
        #         # fw.write(bbox_info)
        #         # fw.write(" ")
        #         fw.write(bbox_label)
        #         fw.write(" ")
        #         fw.write("0\n")
        # else:
        #     with open(out_path+img_id[:-4]+'.txt',"a") as (fw):
        #         for k in range(len(bbox_info)):
        #             fw.write(str(int(bbox_info[k])))
        #             fw.write(" ")
        #         # fw.write(bbox_info)
        #         # fw.write(" ")
        #         fw.write(bbox_label)
        #         fw.write(" ")
        #         fw.write("0\n")

# aitod json to txt

# ann_path = '/data/2_data_server/cv_data/ai_todv2/aitodv2_train.json'
# out_path = '/data/2_data_server/cv_data/ai_todv2/train/annfiles/'

# # ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
# label_test = []

# for i in glob(ann_path):
#     # print(i)
#     with open(i) as f:
#         json_data = json.load(f)
   
#     img_id = json_data['features'][0]['properties']['image_id']
#     for j in range(len(json_data['features'])):

#         bbox_info = json_data['features'][j]['properties']['object_imcoords']
#         bbox_info = ast.literal_eval(bbox_info)
#         bbox_info = list(bbox_info)
        
#         bbox_label = json_data['features'][j]['properties']['type_name'].replace(" ","-")
#         # bbox_id = json_data['features'][j]['properties']['type_id']
#         # if label_test == []:
#         #     # label_test.append(bbox_id)
#         #     label_test.append(bbox_label)
#         # if bbox_label not in label_test:
#         #     # label_test.append(bbox_id)
#         #     label_test.append(bbox_label)

       
#         # first [:4] 지운 후
#         if j == 0:
#             with open(out_path+img_id[:-4]+'.txt',"w") as (fw):
#                 for k in range(len(bbox_info)):
#                     fw.write(str(int(bbox_info[k])))
#                     fw.write(" ")
#                 # fw.write(bbox_info)
#                 # fw.write(" ")
#                 fw.write(bbox_label)
#                 fw.write(" ")
#                 fw.write("0\n")
#         else:
#             with open(out_path+img_id[:-4]+'.txt',"a") as (fw):
#                 for k in range(len(bbox_info)):
#                     fw.write(str(int(bbox_info[k])))
#                     fw.write(" ")
#                 # fw.write(bbox_info)
#                 # fw.write(" ")
#                 fw.write(bbox_label)
#                 fw.write(" ")
#                 fw.write("0\n")




# min,max 출력
# ann_path = '/data/2_data_server/cv_data/arirang_split/train_ms/annfiles/'

# num_min = 100000
# num_max = 0

# num_total = 0
# for i in glob(ann_path+'*.txt'):
#     # print(i)
#     num_lines = sum(1 for line in open(i))

#     num_min = min(num_lines, num_min)
#     num_max = max(num_lines, num_max)
#     if num_max == 1891:
#         print(i)
#         exit()

# print(num_min,num_max)


# # gt 개수
# ann_path = '/data/2_data_server/cv_data/arirang_split/val_ms/annfiles/'

# # CLASSES = ('small-ship', 'large-ship', 'civilian-aircraft', 'military-aircraft', 'small-car', 'bus', 'truck', 'train', 'crane', 'bridge', 
# #             'oil-tank', 'dam', 'outdoor-playground', 'helipad', 'roundabout', 'indoor-playground','helicopter','individual-container','grouped-container','swimming-pool','etc')

# CLASSES = ('small-ship', 'large-ship', 'civilian-aircraft', 'military-aircraft', 'small-car', 'bus', 'truck', 'train')


# label_cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# cls_map = {c: i
#             for i, c in enumerate(CLASSES)
#             }

# for i in glob(ann_path+'*.txt'):
#     # print(i)
#     f = open(i,"r")
#     lines = f.readlines()
#     for line in lines:
#         label = line.split()
#         cls_name = label[8]
#         if cls_name == 'military-aircraft':
#             print(i)
#             exit()
#         # label = cls_map[cls_name]
#         # label_cnt[label] = label_cnt[label] + 1

# # print(label_cnt)


##### size check
# ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
# # out_path = '/data/2_data_server/cv_data/arirang/val/annfiles/'

# # ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
# label_test = []

# c1 = 0
# c2 = 0
# c3 = 0
# c4 = 0
# c5 = 0
# c6 = 0
# c7 = 0
# c8 = 0
# c9 = 0
# c10 = 0
# c11 = 0
# c12 = 0
# c13 = 0
# c14 = 0
# c15 = 0
# c16 = 0
# c17 = 0
# c18 = 0
# c19 = 0
# c20 = 0
# c21 = 0

# c1_num = 0
# c2_num = 0
# c3_num = 0
# c4_num = 0
# c5_num = 0
# c6_num = 0
# c7_num = 0
# c8_num = 0
# c9_num = 0
# c10_num = 0
# c11_num = 0
# c12_num = 0
# c13_num = 0
# c14_num = 0
# c15_num = 0
# c16_num = 0
# c17_num = 0
# c18_num = 0
# c19_num = 0
# c20_num = 0
# c21_num = 0

# for i in glob(ann_path+'*.json'):
#     # print(i)
#     with open(i) as f:
#         json_data = json.load(f)
  
#     for j in range(len(json_data['features'])):

#         bbox_info = json_data['features'][j]['properties']['object_imcoords']
#         bbox_info = ast.literal_eval(bbox_info)
        
#         poly = np.array(bbox_info,dtype=np.float32)
    
        
#         poly = poly2obb_np(poly)
        
#         if poly is not None:
#             w = poly[2]
#             h = poly[3]
#             area = w*h
#         # area = 
#             bbox_label = json_data['features'][j]['properties']['type_name'].replace(" ","-")

#             if bbox_label =="small-ship":
#                 c1 += 1
#                 c1_num += area
#             if bbox_label =="large-ship":
#                 c2 += 1
#                 c2_num += area
#             if bbox_label =="civilian-aircraft":
#                 c3 += 1
#                 c3_num += area
#             if bbox_label =="military-aircraft":
#                 c4 += 1
#                 c4_num += area
#             if bbox_label =="small-car":
#                 c5 += 1
#                 c5_num += area
#             if bbox_label =="bus":
#                 c6 += 1
#                 c6_num += area
#             if bbox_label =="truck":
#                 c7 += 1
#                 c7_num += area
#             if bbox_label =="train":
#                 c8 += 1
#                 c8_num += area
#             if bbox_label =="crane":
#                 c9 += 1
#                 c9_num += area
#             if bbox_label =="bridge":
#                 c10 += 1
#                 c10_num += area
#             if bbox_label =="oil-tank":
#                 c11 += 1
#                 c11_num += area
#             if bbox_label =="dam":
#                 c12 += 1
#                 c12_num += area
#             if bbox_label =="outdoor-playground":
#                 c13 += 1
#                 c13_num += area
#             if bbox_label =="helipad":
#                 c14 += 1
#                 c14_num += area
#             if bbox_label =="roundabout":
#                 c15 += 1
#                 c15_num += area
#             if bbox_label =="indoor-playground":
#                 c16 += 1
#                 c16_num += area
#             if bbox_label =="helicopter":
#                 c17 += 1
#                 c17_num += area
#             if bbox_label =="individual-container":
#                 c18 += 1
#                 c18_num += area
#             if bbox_label =="grouped-container":
#                 c19 += 1
#                 c19_num += area
#             if bbox_label =="swimming-pool":
#                 c20 += 1
#                 c20_num += area 

# print("c1------")
# print(c1,c1_num)
# print("------")
# print("c2------")
# print(c2,c2_num)
# print("------")
# print("c3------")
# print(c3,c3_num)
# print("------")
# print("c4------")
# print(c4,c4_num)
# print("------")
# print("c5------")
# print(c5,c5_num)
# print("------")
# print("c6------")
# print(c6,c6_num)
# print("------")
# print("c7------")
# print(c7,c7_num)
# print("------")
# print("c8------")
# print(c8,c8_num)
# print("------")
# print("c9------")
# print(c9,c9_num)
# print("------")
# print("c10------")
# print(c10,c10_num)
# print("------")
# print("c11------")
# print(c11,c11_num)
# print("------")
# print("c12------")
# print(c12,c12_num)
# print("------")
# print("c13------")
# print(c13,c13_num)
# print("------")
# print("c14------")
# print(c14,c14_num)
# print("------")
# print("c15------")
# print(c15,c15_num)
# print("------")
# print("c16------")
# print(c16,c16_num)
# print("------")
# print("c17------")
# print(c17,c17_num)
# print("------")
# print("c18------")
# print(c18,c18_num)
# print("------")
# print("c19------")
# print(c19,c19_num)
# print("------")
# print("c20------")
# print(c20,c20_num)
# print("------")
