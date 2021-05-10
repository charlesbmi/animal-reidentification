"""Adapted from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os.path
import pathlib
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None):
        """Set the path for images and annotations.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and animal ID)."""
        coco = self.coco
        ann_id = self.ids[index]
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Return image and animal identifier
        return image, target

    def __len__(self):
        return len(self.ids)


def get_loader(root, json, transform, batch_size, shuffle=True, num_workers=4):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    coco = CocoDataset(root=root,
            json=json,
            transform=transform)

    # Data loader for COCO dataset
    # This will return (images, animal-ID) for each iteration.
    # images: a tensor of shape (batch_size, 3, INPUT_SIZE, INPUT_SIZE).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)
    return data_loader


def main():
    # Test the class
    import argparse
    parser = argparse.ArgumentParser(description='test data_loader')
    parser.add_argument('-i', '--images', type=pathlib.Path,
            help='folder with images')
    parser.add_argument('-j', '--json', type=pathlib.Path,
            help='Annotations JSON file in COCO-format')
    args = parser.parse_args()

    transforms = torchvision.transforms.ToTensor()
    batch_size = 32
    shuffle=True
    data_loader = get_loader(args.images, args.json, transforms, batch_size, shuffle)

    return


if __name__ == '__main__':
    main()
