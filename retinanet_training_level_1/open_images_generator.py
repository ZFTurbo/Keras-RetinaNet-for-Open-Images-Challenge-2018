"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import json
import os
import warnings
import random
import numpy as np

from keras_retinanet.preprocessing.generator import Generator
# from keras_retinanet.utils.image import read_image_bgr
from a00_utils_and_constants import read_image_bgr_fast, OUTPUT_PATH, random_intensity_change1


def get_labels(metadata_dir, version='v4'):
    csv_file = 'class-descriptions-boxable-level-1.csv'

    boxable_classes_descriptions = os.path.join(metadata_dir, csv_file)
    id_to_labels = {}
    cls_index    = {}

    i = 0
    with open(boxable_classes_descriptions) as f:
        for row in csv.reader(f):
            # make sure the csv row is not empty (usually the last one)
            if len(row):
                label       = row[0]
                description = row[1].replace("\"", "").replace("'", "").replace('`', '')
                id_to_labels[i]  = description
                cls_index[label] = i
                i += 1

    return id_to_labels, cls_index


def get_image_sizes(subset):
    import pandas as pd
    sizes = pd.read_csv(OUTPUT_PATH + subset + '_image_params.csv')
    ret = dict()
    ids = sizes['id'].values
    ws = sizes['width'].values
    ht = sizes['height'].values
    for i in range(len(ids)):
        ret[ids[i]] = (int(ws[i]), int(ht[i]))
    return ret


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version='v4'):
    validation_image_ids = {}

    if version == 'v4':
        annotations_path = os.path.join(metadata_dir, '{}-annotations-bbox-level-1.csv'.format(subset))
    elif version == 'challenge2018':
        validation_image_ids_path = os.path.join(metadata_dir, 'challenge-2018-image-ids-valset-od.csv')

        with open(validation_image_ids_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=['ImageID'])
            reader.next()
            for line, row in enumerate(reader):
                image_id = row['ImageID']
                validation_image_ids[image_id] = True

        annotations_path = os.path.join(metadata_dir, 'challenge-2018-train-annotations-bbox.csv')
    else:
        annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    fieldnames = ['ImageID', 'Source', 'LabelName', 'Confidence',
                  'XMin', 'XMax', 'YMin', 'YMax',
                  'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']

    id_annotations = dict()
    with open(annotations_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        next(reader)

        images_sizes = get_image_sizes(subset)
        for line, row in enumerate(reader):
            frame = row['ImageID']
            img_id = row['ImageID']
            class_name = row['LabelName']

            if version == 'challenge2018':
                if subset == 'train':
                    if frame in validation_image_ids:
                        continue
                elif subset == 'validation':
                    if frame not in validation_image_ids:
                        continue
                else:
                    raise NotImplementedError('This generator handles only the train and validation subsets')

            if version == 'challenge2018':
                # We recommend participants to use the provided subset of the training set as a validation set.
                # This is preferable over using the V4 val/test sets, as the training set is more densely annotated.
                img_path = os.path.join(main_dir, 'train', frame[:3], frame + '.jpg')
            else:
                if subset == 'validation':
                    img_path = os.path.join(main_dir, 'validation', frame + '.jpg')
                else:
                    img_path = os.path.join(main_dir, subset, frame[:3], frame + '.jpg')

            if not os.path.isfile(img_path):
                continue

            try:
                width, height = images_sizes[frame]
            except:
                print('Image read error: {}'.format(frame))
                continue

            if class_name == '':
                if img_id in id_annotations:
                    print('Strange duplicate {}'.format(img_id))
                    exit()
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': []}
                continue

            if class_name not in cls_index:
                continue

            cls_id = cls_index[class_name]

            x1 = float(row['XMin'])
            x2 = float(row['XMax'])
            y1 = float(row['YMin'])
            y2 = float(row['YMax'])

            x1_int = int(round(x1 * width))
            x2_int = int(round(x2 * width))
            y1_int = int(round(y1 * height))
            y2_int = int(round(y2 * height))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if y2_int == y1_int:
                warnings.warn('filtering line {}: rounding y2 ({}) and y1 ({}) makes them equal'.format(line, y2, y1))
                continue

            if x2_int == x1_int:
                warnings.warn('filtering line {}: rounding x2 ({}) and x1 ({}) makes them equal'.format(line, x2, x1))
                continue

            annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

            if img_id in id_annotations:
                annotations = id_annotations[img_id]
                annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}
    return id_annotations


def get_class_index_arrays(number_of_classes, annotation_cache_json):
    classes = dict()
    f = open(annotation_cache_json, 'r')
    annotations = json.loads(f.read())

    classes['empty'] = []
    for c in range(number_of_classes):
        classes[c] = []

    for id in annotations:
        if 'boxes' in annotations[id]:
            if len(annotations[id]['boxes']) == 0:
                classes['empty'].append(id)
            for box in annotations[id]['boxes']:
                c = box['cls_id']
                classes[c].append(id)

    return classes


class OpenImagesGenerator(Generator):
    def __init__(
            self, main_dir, subset, version='v4',
            labels_filter=None, annotation_cache_dir='.',
            fixed_labels=False,
            **kwargs
    ):

        if subset == 'validation':
            self.base_dir = os.path.join(main_dir, 'images', 'validation')
        else:
            self.base_dir = os.path.join(main_dir, 'images', subset)

        metadata_dir          = OUTPUT_PATH + 'level_1_files/'
        annotation_cache_json = os.path.join(metadata_dir, subset + '_level_1.json')

        self.id_to_labels, cls_index = get_labels(metadata_dir, version=version)
        print('Labels length: {}'.format(len(cls_index)))
        # print(self.id_to_labels)
        # print(cls_index)
        # exit()

        if os.path.exists(annotation_cache_json):
            with open(annotation_cache_json, 'r') as f:
                self.annotations = json.loads(f.read())
        else:
            self.annotations = generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index, version=version)
            json.dump(self.annotations, open(annotation_cache_json, "w"))

        if labels_filter is not None:
            self.id_to_labels, self.annotations = self.__filter_data(labels_filter, fixed_labels)

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])
        self.image_id_to_id = dict([(k, i) for i, k in enumerate(self.annotations)])
        self.class_index_array = get_class_index_arrays(len(cls_index), annotation_cache_json)
        self.group_method = 'random'
        self.subset = subset

        super(OpenImagesGenerator, self).__init__(**kwargs)


    def __filter_data(self, labels_filter, fixed_labels):
        """
        If you want to work with a subset of the labels just set a list with trainable labels
        :param labels_filter: Ex: labels_filter = ['Helmet', 'Hat', 'Analog television']
        :param fixed_labels: If fixed_labels is true this will bring you the 'Helmet' label
        but also: 'bicycle helmet', 'welding helmet', 'ski helmet' etc...
        :return:
        """

        labels_to_id = dict([(l, i) for i, l in enumerate(labels_filter)])

        sub_labels_to_id = {}
        if fixed_labels:
            # there is/are no other sublabel(s) other than the labels itself
            sub_labels_to_id = labels_to_id
        else:
            for l in labels_filter:
                label = str.lower(l)
                for v in [v for v in self.id_to_labels.values() if label in str.lower(v)]:
                    sub_labels_to_id[v] = labels_to_id[l]

        filtered_annotations = {}
        for k in self.annotations:
            img_ann = self.annotations[k]

            filtered_boxes = []
            for ann in img_ann['boxes']:
                cls_id = ann['cls_id']
                label = self.id_to_labels[cls_id]
                if label in sub_labels_to_id:
                    ann['cls_id'] = sub_labels_to_id[label]
                    filtered_boxes.append(ann)

            if len(filtered_boxes) > 0:
                filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

        id_to_labels = dict([(labels_to_id[k], k) for k in labels_to_id])
        return id_to_labels, filtered_annotations

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return len(self.id_to_labels)

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return self.id_to_labels[label]

    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id_to_image_id[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    def image_path(self, image_index):
        type = os.path.basename(self.base_dir)
        up = os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), type)
        id = self.id_to_image_id[image_index]
        if type == 'train':
            path = os.path.join(up, id[:3], id + '.jpg')
        else:
            path = os.path.join(up, id + '.jpg')
        return path

    def load_image(self, image_index):
        # return read_image_bgr(self.image_path(image_index))
        return read_image_bgr_fast(self.image_path(image_index))

    def load_annotations(self, image_index):
        image_annotations = self.annotations[self.id_to_image_id[image_index]]

        labels = image_annotations['boxes']
        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))
        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes

    def group_images(self):
        classes = list(range(self.num_classes())) + ['empty']
        self.groups = []
        while 1:
            if len(self.groups) > 100000:
                break
            self.groups.append([])
            for i in range(self.batch_size):
                while 1:
                    random_class = random.choice(classes)
                    # print(random_class, len(self.class_index_array[random_class]))
                    if len(self.class_index_array[random_class]) > 0:
                        random_image = random.choice(self.class_index_array[random_class])
                        break
                random_image_index = self.image_id_to_id[random_image]
                self.groups[-1].append(random_image_index)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """

        if self.subset != 'validation':
            # random color change
            image = random_intensity_change1(image, -30, 30, True)

        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                self.group_images()
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)