import json
import numpy as np
import os

np.random.seed(21)

# load the annotations
BOX_ANNOTATION_FILE = '/media/data/ComputerVisionCourse/zebragiraffe/labels.json'
with open(BOX_ANNOTATION_FILE) as f:
    data_orig = json.load(f)
f.close()

# get zebra annotations (excludes giraffe)
zebra_annotations = [ann for ann in data_orig['annotations'] if ann['category_id'] == 1]

# get the zebra names from the annotations, 6286 zebra annotations, 639 giraffe annotations
zebra_names = [ann['name'] for ann in zebra_annotations]

# get the unique individual zebra names and counts
zebra_names_counts = np.unique(zebra_names, return_counts=True)

#  tells you that there's 876 zebra names with 1 sighting, 346 zebra names with 2 sightings, etc
print(np.unique(np.unique(zebra_names, return_counts=True)[1], return_counts=True))

# group zebra ids by single sighting, two sightings, or more than two
zebra_names_one = zebra_names_counts[0][np.where(zebra_names_counts[1] == 1)]  # zebra with just one image
zebra_names_two = zebra_names_counts[0][np.where(zebra_names_counts[1] == 2)]
zebra_names_over_two = zebra_names_counts[0][np.where(zebra_names_counts[1] > 2)]

# split each group into 0.7 train / 0.1 val / 0.2 test
trainFrac = 0.7
valFrac = 0.1
testFrac = 0.2  # just assumed to be the remainder of the data after train and val
# split ones
rand_split_inds = np.random.permutation(np.arange(len(zebra_names_one))).astype(int)
trainValDivide = int(np.floor(trainFrac * len(zebra_names_one)))
valTestDivide = trainValDivide + int(np.floor(valFrac * len(zebra_names_one)))
zebra_one_train = [zebra_names_one[a] for a in rand_split_inds[:trainValDivide]]
zebra_one_val = [zebra_names_one[a] for a in rand_split_inds[trainValDivide:valTestDivide]]
zebra_one_test = [zebra_names_one[a] for a in rand_split_inds[valTestDivide:]]

# split twos
rand_split_inds = np.random.permutation(np.arange(len(zebra_names_two))).astype(int)
trainValDivide = int(np.floor(trainFrac * len(zebra_names_two)))
valTestDivide = trainValDivide + int(np.floor(valFrac * len(zebra_names_two)))
zebra_two_train = [zebra_names_two[a] for a in rand_split_inds[:trainValDivide]]
zebra_two_val = [zebra_names_two[a] for a in rand_split_inds[trainValDivide:valTestDivide]]
zebra_two_test = [zebra_names_two[a] for a in rand_split_inds[valTestDivide:]]

# split over twos
rand_split_inds = np.random.permutation(np.arange(len(zebra_names_over_two))).astype(int)
trainValDivide = int(np.floor(trainFrac * len(zebra_names_over_two)))
valTestDivide = trainValDivide + int(np.floor(valFrac * len(zebra_names_over_two)))
zebra_over_two_train = [zebra_names_over_two[a] for a in rand_split_inds[:trainValDivide]]
zebra_over_two_val = [zebra_names_over_two[a] for a in rand_split_inds[trainValDivide:valTestDivide]]
zebra_over_two_test = [zebra_names_over_two[a] for a in rand_split_inds[valTestDivide:]]

# concatenate ids over groups
zebra_train = zebra_one_train + zebra_two_train + zebra_over_two_train
zebra_val = zebra_one_val + zebra_two_val + zebra_over_two_val
zebra_test = zebra_one_test + zebra_two_test + zebra_over_two_test

# for each group, restructure dataset and save out
print(data_orig.keys())
print(data_orig['annotations'][0])

# create training dataset
data_train = {}
data_train['categories'] = data_orig['categories'].copy()
data_train['images'] = data_orig['images'].copy()
data_train['annotations'] = []
for name in zebra_train:
    # append all images this zebra appears in
    [data_train['annotations'].append(ann) for ann in data_orig['annotations'] if ann['name'] == name]

# create validation dataset
data_val = {}
data_val['categories'] = data_orig['categories'].copy()
data_val['images'] = data_orig['images'].copy()
data_val['annotations'] = []
for name in zebra_val:
    # append all images this zebra appears in
    [data_val['annotations'].append(ann) for ann in data_orig['annotations'] if ann['name'] == name]

# create test dataset
data_test = {}
data_test['categories'] = data_orig['categories'].copy()
data_test['images'] = data_orig['images'].copy()
data_test['annotations'] = []
for name in zebra_test:
    # append all images this zebra appears in
    [data_test['annotations'].append(ann) for ann in data_orig['annotations'] if ann['name'] == name]

# end up with 4328 training annotations, 677 validation annotations, and 1681 testing annotations (total 6304 annotations)
totalAnns = len(data_test['annotations']) + len(data_val['annotations']) + len(data_train['annotations'])
print(len(data_train['annotations']))
print(len(data_val['annotations']))
print(len(data_test['annotations']))
print(totalAnns)

# save out the new jsons
new_data_path = '/media/data/ComputerVisionCourse/zebragiraffe/annotations/'
os.makedirs(new_data_path, exist_ok=True) # create directory if needed

with open(new_data_path + 'customSplit_train.json', 'w') as outfile:
    json.dump(data_train, outfile, indent = 4, ensure_ascii = False)
    print('Wrote to:', outfile.name)

with open(new_data_path + 'customSplit_val.json', 'w') as outfile:
    json.dump(data_val, outfile, indent = 4, ensure_ascii = False)
    print('Wrote to:', outfile.name)

with open(new_data_path + 'customSplit_test.json', 'w') as outfile:
    json.dump(data_test, outfile, indent = 4, ensure_ascii = False)
    print('Wrote to:', outfile.name)

#6286
# targ train = 4400
# targ val = 628
# targ test = 1258
# so pretty close
