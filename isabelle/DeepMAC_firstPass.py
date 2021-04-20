import glob
import io
import json
import logging
import os
import random
import warnings

import imageio
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
from six import BytesIO
from skimage import color
from skimage import transform
from skimage import util
from skimage.color import rgb_colors
import tensorflow as tf
from matplotlib.patches import Rectangle

COLORS = ([rgb_colors.cyan, rgb_colors.orange, rgb_colors.pink,
           rgb_colors.purple, rgb_colors.limegreen, rgb_colors.crimson] +
          [(color) for (name, color) in color.color_dict.items()])
random.shuffle(COLORS)

logging.disable(logging.WARNING)


def read_image(path):
    """Read an image and optionally resize it for better plotting."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img, dtype=np.uint8)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def create_detection_map(annotations):
    """Creates a dict mapping IDs to detections."""

    ann_map = {}
    # for image in annotations['images']:
    #   ann_map[image['id']] = image['detections']
    # return ann_map

    for image in annotations['images']:
        # get all the detections for that image
        image_id = image['id']
        # print("image_id")
        # print(image_id)
        detections = []
        for a in annotations['annotations']:
            if a['image_id'] == image_id:
                a['float_bbox'] = a['bbox'].copy()
                bbox = a['segmentation_bbox'].copy()
                # swap the bounding box dimensions to be [x y width height]
                a['bbox'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                detections.append(a)
        # print("DETECTIONS")
        # print(detections)
        ann_map[str(image_id)] = detections
        # print('DICT')
        # print(ann_map[str(image_id)])
        # print("KEYS")
        # print(ann_map.keys())
        # plt.plot(1)
        # plt.show()
    return ann_map


def get_mask_prediction_function(model):
    """Get single image mask prediction function using a model."""

    @tf.function
    def predict_masks(image, boxes):
        height, width, _ = image.shape.as_list()
        batch = image[tf.newaxis]
        boxes = boxes[tf.newaxis]

        detections = model(batch, boxes)
        masks = detections['detection_masks']

        return reframe_box_masks_to_image_masks(masks[0], boxes[0],
                                                height, width)

    return predict_masks


def convert_boxes(boxes):
    xmin, ymin, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ymax = ymin + height
    xmax = xmin + width

    return np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)


# Copied from tensorflow/models
def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
    """Transforms the box masks back to full image masks.
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.
  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
    resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method

    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            denom = max_corner - min_corner
            # Prevent a divide by zero.
            denom = tf.math.maximum(denom, 1e-4)
            transformed_boxes = (boxes - min_corner) / denom
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

        # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
        # is static. This will help us run and test on TPUs.
        resized_crops = tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_indices=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            method=resize_method,
            extrapolation_value=0)
        return tf.cast(resized_crops, box_masks.dtype)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
    return tf.squeeze(image_masks, axis=3)


def plot_image_annotations(image, boxes, masks, darken_image=0.5):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_axis_off()
    image = (image * darken_image).astype(np.uint8)
    ax.imshow(image)

    height, width, _ = image.shape

    num_colors = len(COLORS)
    color_index = 0

    for box, mask in zip(boxes, masks):
        ymin, xmin, ymax, xmax = box
        ymin *= height
        ymax *= height
        xmin *= width
        xmax *= width

        color = COLORS[color_index]
        color = np.array(color)
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        mask = (mask > 0.5).astype(np.float32)
        color_image = np.ones_like(image) * color[np.newaxis, np.newaxis, :]
        color_and_mask = np.concatenate(
            [color_image, mask[:, :, np.newaxis]], axis=2)

        ax.imshow(color_and_mask, alpha=0.5)

        color_index = (color_index + 1) % num_colors

    return ax


# if you haven't already, execute these commands in the command line in order to download the appropriate checkpoint
# curl -o deepmac_1024x1024_coco17.tar.gz http://download.tensorflow.org/models/object_detection/tf2/20210329/deepmac_1024x1024_coco17.tar.gz
# tar -xzf deepmac_1024x1024_coco17.tar.gz
# and make sure you put it in the right location

model = tf.keras.models.load_model('../deepMAC/deepmac_1024x1024_coco17/saved_model')
prediction_function = get_mask_prediction_function(model)

BOX_ANNOTATION_FILE = '../Data/gzgc.coco/annotations/instances_train2020.json'
detection_map = create_detection_map(read_json(BOX_ANNOTATION_FILE))

image_path = '../Data/gzgc.coco/images/train2020/000000000002.jpg'
image_id = os.path.basename(image_path).rstrip('.jpg')
image_id = str(int(image_id))
# print(detection_map.keys())
print("Image ID: " + image_id)
if image_id not in detection_map:
    print(f'Image {image_path} is missing detection data.')
elif len(detection_map[image_id]) == 0:
    print(f'There are no detected objects in the image {image_path}.')
else:
    detections = detection_map[image_id]
    image = read_image(image_path)
    bboxes = np.array([det['bbox'] for det in detections]) # [x, y, width, height]
    print("BBOXES")
    print(bboxes)
    bboxes = convert_boxes(bboxes) # becomes [ymin, xmin, ymax, xmax]
    print("BBOXES Converted")
    print(bboxes)
    print(image.shape)
    f = plt.figure(figsize=(6, 5))
    ax = plt.subplot()
    plt.imshow(image)
    ax.add_patch(Rectangle((bboxes[0][1], bboxes[0][0]),
                            bboxes[0][3]- bboxes[0][1], bboxes[0][2]- bboxes[0][0],
                           linewidth=1, edgecolor='g', facecolor='none'))
    plt.show()
    immm = tf.convert_to_tensor(image)
    bbb = tf.convert_to_tensor(bboxes, dtype=tf.float32)
    masks = prediction_function(immm, bbb)
    # masks = prediction_function(tf.convert_to_tensor(image),
    #                             tf.convert_to_tensor(bboxes, dtype=tf.float32))
    plot_image_annotations(image, bboxes, masks.numpy(), darken_image=0.75)
