import cv2
import os
import numpy as np
import json
import mmcv
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

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

# img_path -> model detect result image
img_path = '/data/2_data_server/cv_data/mmrotate_dota/angle_test/images/mAP_test.png'
ann_path = '/data/2_data_server/cv_data/mmrotate_dota/angle_test/annfiles/P0003__1024__123___0.txt'
out_path = '/data/2_data_server/cv-01/vis/mAP_TEST.png'

gt_bboxes = []
win_name = ''

with open(ann_path) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np_oc(poly)
                        except:  # noqa: E722
                            continue
                        
                        gt_bboxes.append([x, y, w, h, a])

img = mmcv.imread(img_path).astype(np.uint8)
img = mmcv.bgr2rgb(img)
width, height = img.shape[1], img.shape[0]
img = np.ascontiguousarray(img)

fig = plt.figure(win_name, frameon=False)
plt.title(win_name)
canvas = fig.canvas
dpi = fig.get_dpi()
fig.set_size_inches((width + 1e-2) / dpi, (height + 1e-2) / dpi)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = plt.gca()
ax.axis('off')


num_bboxes = len(gt_bboxes)
# gt_bboxes = tuple(gt_bboxes)
draw_rbboxes(ax, gt_bboxes, alpha=0.6, thickness=2)

horizontal_alignment = 'left'

# positions = gt_bboxes[...,:2].astype(np.int32)
# areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
# scales = _get_adaptive_scales(areas)

plt.imshow(img)

stream, _ = canvas.print_to_buffer()
buffer = np.frombuffer(stream, dtype='uint8')
img_rgba = buffer.reshape(height, width, 4)
rgb, alpha = np.split(img_rgba, [3], axis=2)
img = rgb.astype('uint8')
img = mmcv.rgb2bgr(img)

mmcv.imwrite(img, out_path)


