"""Adapted from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""

import torch
import torchvision
import os.path
import pathlib
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TripletZebras(torch.utils.data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None, num_triplets=100*1000, apply_mask=False, apply_mask_bbox=False):
        """Set the path for images and annotations.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
            filter_category (iterable): set of categories to include, e.g. [0, 1]
        """
        self.root = root
        coco = COCO(json)
        self.annotations = coco.anns
        self.images = coco.imgs
        self.mask = apply_mask
        self.mask_bbox = apply_mask_bbox
        
        zebra_annotations = [ann for ann in self.annotations.values() if ann['category_id'] == 1]
        zebra_names = [ann['name'] for ann in zebra_annotations]
        unique_zebra_names, zebra_names_counts = np.unique(zebra_names, return_counts=True) # numpy arrays
        anchors = unique_zebra_names[np.where(zebra_names_counts > 1)]

        # Generate triplets of annotation IDs (keys into self.annotations dictionary)
        triplets = []
        for i in tqdm(np.arange(num_triplets)):
            triplets.append(self.generate_triplet(anchors, unique_zebra_names))
        # Remove duplicates
        triplets = np.unique(triplets, axis=0).tolist()

        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        """Returns triplet of images"""

        assert len(self.triplets[index]) == 3, 'Expected triplet corresponding to anchor, positive, negative'

        anchor_positive_negative = []
        for annotation_id in self.triplets[index]:
            annotation = self.annotations[annotation_id]
            assert annotation['id'] == annotation_id

            image_id = annotation['image_id']
            image_info = self.images[image_id]
            assert image_info['id'] == image_id

            image_fname = image_info['file_name']
            image_path = os.path.join(self.root, image_fname)

            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply segmentation mask
            if self.mask==True:
                mask = mask_util.decode(annotation['maskrcnn_mask_rle'])
                segImage  = np.array(image)
                binaryMask = (mask > 0.5).astype(np.float32)
                segImage[np.where(binaryMask == 0.0)] = 0
                image = Image.fromarray(np.uint8(segImage)).convert('RGB')
                # Crop to bounding box
                image = self.crop_to_bbox(image, annotation['maskrcnn_bbox'])
            
            if self.mask_bbox:
                # Crop to bounding box
                image = self.crop_to_bbox(image, annotation['bbox_tlbr'])
            # Transform to tensor
            if self.transform:
                image = self.transform(image)

            # Save to list
            anchor_positive_negative.append(image)

        return anchor_positive_negative, self.triplets[index]

    def __len__(self):
        return len(self.triplets)

    def generate_triplet(self, anchors, unique_zebra_names):
        anchor_zebra_name = np.random.choice(anchors, replace=True)
        # Get annotations (ie, bounding boxes) associated with this individual
        anchor_all_annotation_ids = [ann_id for (ann_id, ann) in self.annotations.items() if ann['name'] == anchor_zebra_name]

        # Pick 2 bboxes to be anchor and positive
        anchor_annotation_id, positive_annotation_id = np.random.choice(anchor_all_annotation_ids, size=2, replace=False)

        # Pick a zebra individual that is NOT our anchor/positive
        other_zebra_names = np.setdiff1d(unique_zebra_names, anchor_zebra_name)
        neg_zebra_name = np.random.choice(other_zebra_names, replace=False)
        negative_all_annotation_ids = [ann_id for (ann_id, ann) in self.annotations.items() if ann['name'] == neg_zebra_name]
        negative_annotation_id = np.random.choice(negative_all_annotation_ids, replace=False)

        return (anchor_annotation_id, positive_annotation_id, negative_annotation_id)

    def crop_to_bbox(self, image: Image.Image, bbox: tuple):
        # Assume order of bbox from maskrcnn
        x_left, y_top, x_right, y_bottom = bbox
        width = x_right - x_left
        height = y_bottom - y_top
        x_center = (x_left + x_right) / 2
        y_center = (y_top + y_bottom) / 2

        # Crop to a square box, so this doesn't get cut off later
        new_size = max(width, height)
        x_left = round(x_center - (new_size / 2))
        y_top = round(y_center - (new_size / 2))

        cropped_image = torchvision.transforms.functional.crop(image, y_top, x_left, new_size, new_size)

        return cropped_image


def get_loader(root, json, transform, batch_size, shuffle=True, num_workers=4, num_triplets=100*1000, apply_mask=False, apply_mask_bbox=False):
    zebra_triplets = TripletZebras(root=root,
        json=json,
        transform=transform,
        num_triplets=num_triplets,
        apply_mask=apply_mask,
        apply_mask_bbox=apply_mask_bbox
    )

    # Data loader for COCO dataset
    # This will return (images, animal-ID) for each iteration.
    # images: a tensor of shape (batch_size, 3, INPUT_SIZE, INPUT_SIZE).
    data_loader = torch.utils.data.DataLoader(dataset=zebra_triplets,
                batch_size=32,
                shuffle=True,
                num_workers=num_workers)
    
    return data_loader


def main():
    # Example usage of the dataset loader
    # These packages are only necessary for this test, so we import here
    import argparse
    import torchvision.transforms as transforms
    import pandas as pd

    parser = argparse.ArgumentParser(description='test data_loader')
    parser.add_argument('-i', '--images', type=pathlib.Path,
            required=True,
            help='folder with images')
    parser.add_argument('-j', '--json', type=pathlib.Path,
            required=True,
            help='Annotations JSON file in COCO-format')
    parser.add_argument('-s', '--random-seed', type=int,
            default=21,
            help='random seed for consistency')
    parser.add_argument('-n', '--num-triplets', type=int,
            default=1000,
            help='number of triplets to generate')
    parser.add_argument('-o', '--output-csv', type=str,
            default=None,
            help='output path to dump positive/negative')
    parser.add_argument('--num-workers', type=int,
            default=4,
            help='num dataloader workers. https://pytorch.org/docs/stable/data.html')
    parser.add_argument('-m', '--apply_mask', action='store_true',
        default=False,
        help='apply segmentation mask to the image')
    parser.add_argument('-b', '--apply_mask_bbox', action='store_true',
        default=False,
        help='mask out bounding box from the image')
    args = parser.parse_args()
    np.random.seed(args.random_seed)

    transforms = transforms.Compose([
        transforms.Resize([500, 750]),
        transforms.ToTensor(),
    ])
    data_loader = get_loader(
        args.images,
        args.json,
        transforms,
        batch_size=4,
        shuffle=False,
        num_triplets=args.num_triplets,
        apply_mask=args.apply_mask
    )

    # Print single element from the data loader
#     for img1, img2, img3 in data_loader:
#         plt.imshow(img1[0].permute(1, 2, 0))
#         plt.imshow(img2[0].permute(1, 2, 0))
#         plt.imshow(img3[0].permute(1, 2, 0))
#         plt.show()
#         break

    # Dump triplets to csv
    if args.output_csv:
        triplets = pd.DataFrame(data_loader.dataset.triplets, columns=['anchor', 'positive', 'negative'])
        triplets.to_csv(args.output_csv, index=False)

    return


if __name__ == '__main__':
    main()
