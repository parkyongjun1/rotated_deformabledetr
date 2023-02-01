import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil

input_path = "/data/2_data_server/cv_data/ai_todv2/AI-TOD/test/images"
output_path = "/data/2_data_server/cv_data/ai_todv2/AI-TOD/test/label"

f = open('/data/2_data_server/cv_data/ai_todv2/AI-TOD/annotations/aitod_test_v1.json')
data = json.load(f)
f.close()

file_names = []


def load_images_from_folder(folder):
  for filename in os.listdir(folder):
        file_names.append(filename)
        

load_images_from_folder(input_path)



def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None

def get_img(filename):
  for img in data['images']:
    if img['file_name'] == filename:
      return img

for filename in file_names:
  # Extracting image 
  img = get_img(filename)
  img_id = img['id']
  img_w = img['width']
  img_h = img['height']

  img_ann = get_img_ann(img_id)

  if img_ann:
    file_object = open(f"{output_path}/{filename[:-4]}.txt", "a+")
    for ann in img_ann:
        current_category = ann['category_id'] - 1 # As yolo format labels start from 0 
        current_bbox = ann['bbox']
        
        x0 = current_bbox[0]
        y0 = current_bbox[1]
        w = current_bbox[2]
        h = current_bbox[3]
        
        x1 = x0 + w
        y1 = y0 + h
        # # Finding midpoints
        # x_centre = (x0 + x1)/2
        # y_centre = (y0 + y1)/2
        
        # w = x1-x0
        # h = y1-y0

        
        # Limiting upto fix number of decimal places
        x0 = format(x0, '.6f')
        y0 = format(y0, '.6f')
        x1 = format(x1, '.6f')
        y1 = format(y1, '.6f')
        cls_name =  data["categories"][current_category]["name"]
    
        # Writing current object 
        file_object.write(f"{x0} {y0} {x1} {y1} {cls_name}\n")
       
    file_object.close()

   