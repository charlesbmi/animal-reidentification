import json
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle


def read_json(path):
    with open(path) as f:
        return json.load(f)


BOX_ANNOTATION_FILE = '../Data/gzgc.coco/annotations/instances_train2020.json'
data = read_json(BOX_ANNOTATION_FILE)
# want the bounding boxes for each image
print(data.keys())

# detection_map = create_detection_map(read_json(BOX_ANNOTATION_FILE))
ann_id = 101  # id-1

# TO DO: figure out how to pull and plot all the bounding boxes for a given image
image_id = int(data['annotations'][ann_id]['image_id'] - 1)

print("IMAGE")
print(data['images'][image_id])
print("ANNOTATIONS")
print(data['annotations'][ann_id])

file_name = data['images'][image_id]['file_name']
image_path = '../Data/gzgc.coco/images/train2020/' + file_name
im_name = os.path.basename(image_path).rstrip('.jpg')
print(im_name)
# bbox is in [tl_col, tl_row, width, height]
bbox = data['annotations'][ann_id]['bbox']
print(bbox)
# read image using PIL:
I = Image.open(image_path)
# convert to numpy array:
I = np.asarray(I)
f = plt.figure(figsize=(6, 5))
ax = plt.subplot()
plt.imshow(I)
ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                       linewidth=1, edgecolor='g', facecolor='none'))
ax.set_title('im: ' + im_name + ', ann id: ' + str(ann_id))
plt.show()
ann_map = {}
"""Creates a dict mapping IDs to detections."""
# for every annotated image
# add to a dictionary - but the format is unclear. What should ['detections'] look like?
# formatting information: https://github.com/visipedia/iwildcam_comp
# image{
#   'id': str,
#   'max_detection_conf': float,
#   'detections':[detection]

# detection{
#   # bounding boxes are in normalized, floating-point coordinates, with the origin at the upper-left
#   'bbox' : [x, y, width, height],
#   # note that the categories returned by the detector are not the categories in the WCS dataset
#   'category': str,
#   'conf': float

# so detections is a list of detections for that image
for image in data['images']:
    # get all the detections for that image
    image_id = image['id']
    detections = []
    for a in data['annotations']:
        if a['image_id'] == image_id:
            detections.append(a)
    ann_map[image['id']] = image['detections']
