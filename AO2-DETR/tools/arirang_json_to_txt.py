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
out_path = '/data/2_data_server/cv_data/arirang/val/annfiles/'

# ann_path = '/data/2_data_server/cv_data/arirang/validate_objects_labeling_json/'
label_test = []

# for i in glob(ann_path+'*.json'):
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


# gt 개수
ann_path = '/data/2_data_server/cv_data/arirang_split/train_ms/annfiles/'

CLASSES = ('small-ship', 'large-ship', 'civilian-aircraft', 'military-aircraft', 'small-car', 'bus', 'truck', 'train', 'crane', 'bridge', 
            'oil-tank', 'dam', 'outdoor-playground', 'helipad', 'roundabout', 'indoor-playground','helicopter','individual-container','grouped-container','swimming-pool','etc')

label_cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cls_map = {c: i
            for i, c in enumerate(CLASSES)
            }

for i in glob(ann_path+'*.txt'):
    # print(i)
    f = open(i,"r")
    lines = f.readlines()
    for line in lines:
        label = line.split()
        cls_name = label[8]
        label = cls_map[cls_name]
        label_cnt[label] = label_cnt[label] + 1

print(label_cnt)
