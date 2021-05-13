"""Adapted from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""

import torch
import os.path
import pathlib
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TripletZebras(torch.utils.data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None, num_triplets=100*1000):
        """Set the path for images and annotations.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
            filter_category (iterable): set of categories to include, e.g. [0, 1]
        """
        self.root = root
        coco = COCO(json)
        self.annotations = list(coco.anns.values())
        self.images = list(coco.imgs.values())

        zebra_annotations = [ann for ann in self.annotations if ann['category_id'] == 1]
        zebra_names = [ann['name'] for ann in zebra_annotations]
        unique_zebra_names = np.unique(zebra_names)
        zebra_names_counts = np.unique(zebra_names, return_counts=True)
        anchors = zebra_names_counts[0][np.where(zebra_names_counts[1] > 1)]

        triplets = []
        for i in tqdm(np.arange(num_triplets)):
            triplets.append(self.generate_triplet(anchors, unique_zebra_names))
        triplets = list(set(triplets))

        self.triplets = triplets
        self.transform = transform

    def __getitem__(self, index):
        """Returns triplet of images"""
        
        anchor_path, positive_path, negative_path = self.triplets[index]
        
        img1 = Image.open(os.path.join(self.root, anchor_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.root, positive_path)).convert('RGB')
        img3 = Image.open(os.path.join(self.root, negative_path)).convert('RGB')
        
        # TO DO: apply mask here
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

            
        # Return image and animal identifier
        return (img1, img2, img3)

    def __len__(self):
        return len(self.triplets)

    def generate_triplet(self, anchors, unique_zebra_names):
        anchor_zebra_name = np.random.choice(anchors, replace=True)
        anchor_img_ids = [ann['image_id'] for ann in self.annotations if ann['name']==anchor_zebra_name]
        anchor_file_names = [img['file_name'] for img in self.images if img['id'] in anchor_img_ids]

        anchor_path = np.random.choice(anchor_file_names, replace=False)

        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = np.random.choice(anchor_file_names, replace=False)

        neg_zebra_name = anchor_zebra_name
        while neg_zebra_name == anchor_zebra_name:
            neg_zebra_name = np.random.choice(unique_zebra_names, replace=False)

        neg_img_ids = [ann['image_id'] for ann in self.annotations if ann['name']==neg_zebra_name]
        neg_file_names = [img['file_name'] for img in self.images if img['id'] in neg_img_ids]
        negative_path = np.random.choice(neg_file_names, replace=False)

        return (anchor_path, positive_path, negative_path)

def get_loader(root, json, transform, batch_size, shuffle=True, num_workers=4):
    zebra_triplets = TripletZebras(root=root,
        json=json,
        transform=transform
    )

    # Data loader for COCO dataset
    # This will return (images, animal-ID) for each iteration.
    # images: a tensor of shape (batch_size, 3, INPUT_SIZE, INPUT_SIZE).
    data_loader = torch.utils.data.DataLoader(dataset=zebra_triplets,
                batch_size=32,
                shuffle=True,
                num_workers=4)
    
    return data_loader


def main():
    # Example usage of the dataset loader
    # These packages are only necessary for this test, so we import here
    import argparse
    import torchvision.transforms as transforms

    parser = argparse.ArgumentParser(description='test data_loader')
    parser.add_argument('-i', '--images', type=pathlib.Path,
            required=True,
            help='folder with images')
    parser.add_argument('-j', '--json', type=pathlib.Path,
            required=True,
            help='Annotations JSON file in COCO-format')
    args = parser.parse_args()

    transforms = transforms.Compose([
        transforms.Resize([500, 750]),
        transforms.ToTensor(),
    ])
    data_loader = get_loader(args.images, args.json, transforms, batch_size=4, shuffle=False)

    # Print single element from the data loader
#     for img1, img2, img3 in data_loader:
#         plt.imshow(img1[0].permute(1, 2, 0))
#         plt.imshow(img2[0].permute(1, 2, 0))
#         plt.imshow(img3[0].permute(1, 2, 0))
#         plt.show()
#         break

    return


if __name__ == '__main__':
    main()
