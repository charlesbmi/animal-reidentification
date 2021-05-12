"""Adapted from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""

import torch
import os.path
import pathlib
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(torch.utils.data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None, filter_category=None):
        """Set the path for images and annotations.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
            filter_category (iterable): set of categories to include, e.g. [0, 1]
        """
        self.root = root
        self.coco = COCO(json)
        # Store a static list of annotation IDs to index into
        self.annotation_ids = list(self.coco.anns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one animal sighting (image and animal name)."""
        # index into the annotations
        annotation_id = self.annotation_ids[index]

        # Load the image
        image_id = self.coco.anns[annotation_id]['image_id']
        image_path = self.coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load the animal ID
        animal_name = self.coco.anns[annotation_id]['name']
        # Note: this assumes that name-numbers are unique, which may not be
        # true across multiple species
        individual_animal_id = int(animal_name.split('_')[-1])

        # Return image and animal individual ID
        return image, individual_animal_id

    def __len__(self):
        return len(self.annotation_ids)


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
    for image, individual_animal_id in data_loader:
        print(f'first batch image shape: {image.shape}')
        print(f'first batch animal_name: {individual_animal_id}')
        break

    return


if __name__ == '__main__':
    main()
