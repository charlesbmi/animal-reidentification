import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import geopandas
from pycocotools.coco import COCO

import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Find geographic diversity of images')
    parser.add_argument('-j', '--json', type=pathlib.Path,
            default='/media/data/ComputerVisionCourse/zebragiraffe/annotations/customSplit_train.json',
            help='Annotations JSON file in COCO-format')
    parser.add_argument('-c', '--category-id-list', type=int,
            nargs='+',
            default=[1],
            help='Which animal categories to include')
    parser.add_argument('-m', '--meters-per-degree', type=float,
            default=111320,
            help='Rough conversion of latitude/longitu degrees to meters')
    parser.add_argument('-t', '--threshold-degrees', type=float,
            default=1e-3,
            help='Threshold (in lat/lon degrees) of distance to consider "different place"')

    return parser.parse_args()


def main():
    args  = parse_args()

    # Load in COCO-format annotations
    coco = COCO(args.json)

    # Filter for zebras
    # loop over standard annotations
    zebra_ann_ids = coco.getAnnIds(catIds=args.category_id_list)

    # Collect annotations into Dataframe for easier analysis
    annotations_list = []

    # Grab geographic info for each annotation
    for ann_id in zebra_ann_ids:
        ann = coco.anns[ann_id]
        zebra_name = ann['name']
        img_id = ann['image_id']
        img_info = coco.imgs[img_id]

        # Weird encoding of negative NaN values
        lat = float(img_info['gps_lat_captured'])
        if lat == -1:
            lat = np.nan
        lon = float(img_info['gps_lon_captured'])
        if lon == -1:
            lon = np.nan

        # Collect in list to turn into a dataframe
        annotations_list.append({
            'image_id': img_id,
            'annotation_id': ann_id,
            'zebra_name': zebra_name,
            'latitude': lat,
            'longitude': lon,
        })

    df = pd.DataFrame(annotations_list)

    # Convert degrees to meters (alternatively, could use pint units package)
    df[['latitude', 'longitude']] = df[['latitude', 'longitude']] * args.meters_per_degree

    # gdf = geopandas.GeoDataFrame(
    #     df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))

    # Make a violin plot of these standard-deviations...

    args.threshold_degrees = 1e-3  # 111 meters
    threshold_same_location_m = args.threshold_degrees * args.meters_per_degree,

    geo_spread = df.groupby('zebra_name')[['latitude', 'longitude']].std()
    geo_spread = geo_spread.dropna('index')

    # Plot some figures
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.boxplot(data=geo_spread, ax=ax)
    ax.set_title(f'How geographically diverse is each zebra\'s sightings?\nSpread of sigma for n={len(geo_spread)} individuals (that had >=2 sightings with lat/lon data).')
    ax.set_ylabel('std (meters)')
    ax.axhline(
        threshold_same_location_m,
        color='black',
        linestyle='dashed',
        label=f'{args.threshold_degrees} deg',
    )
    ax.set_yscale('log')
    ax.legend()

    print('Number of zebras with at least threshold lat/lon standard deviation:')
    print((geo_spread > 111).sum())
    print('Number of zebras with that surpass the threshold for either lat OR lon:',
          (geo_spread > 111).any('columns').sum())


    plt.ion()
    plt.pause(0.001)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
