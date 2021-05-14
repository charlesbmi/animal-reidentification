import json
import numpy as np
import pycocotools.mask as mask_util

with open('../../Data/gzgc.coco/masks/instances_train2020_maskrcnn.json') as f:
  data_train = json.load(f)
f.close()

print(mask_util.decode(data_train['annotations'][0]['maskrcnn_mask_rle']))