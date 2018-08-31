# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_utils_and_constants import *


def get_empty_df(negative_samples):
    neg_samp = pd.DataFrame(negative_samples, columns=['ImageID'])
    neg_samp['Source'] = 'freeform'
    neg_samp['LabelName'] = ''
    neg_samp['Confidence'] = 1.0
    neg_samp['XMin'] = ''
    neg_samp['XMax'] = ''
    neg_samp['YMin'] = ''
    neg_samp['YMax'] = ''
    neg_samp['IsOccluded'] = 0
    neg_samp['IsTruncated'] = 0
    neg_samp['IsGroupOf'] = 0
    neg_samp['IsDepiction'] = 0
    neg_samp['IsInside'] = 0
    return neg_samp


def create_level1_files():
    out_dir = OUTPUT_PATH + 'level_1_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    remove_group_of = True

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-1.csv', 'w')
    for l in LEVEL_1_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    negative_sample_classes = [d2[f] for f in ['Armadillo', 'Axe', 'Balance beam', 'Band-aid', 'Banjo', 'Bomb', 'Bottle opener',
                               'Bowling equipment', 'Calculator', 'Can opener', 'Cantaloupe', 'Cassette deck',
                               'Cat furniture', 'Chainsaw', 'Cheese', 'Chime', 'Chisel', 'Closet',
                               'Cocktail shaker', 'Cooking spray', 'Cream', 'Diaper', 'Dishwasher', 'Drill',
                               'Eraser', 'Face powder', 'Facial tissue holder', 'Fax', 'Flying disc', 'Grinder',
                               'Hair dryer', 'Hair spray', 'Hammer', 'Hand dryer', 'Harmonica', 'Heater',
                               'Hedgehog', 'Hiking equipment', 'Hippopotamus', 'Horizontal bar', 'Human body',
                               'Humidifier', 'Indoor rower', 'Ipod', 'Isopod', 'Jacuzzi', 'Koala', 'Ladle',
                               'Lipstick', 'Magpie', 'Maracas', 'Milk', 'Mixing bowl', 'Panda', 'Paper cutter',
                               'Parking meter', 'Pencil case', 'Pencil sharpener', 'Perfume', 'Pizza cutter',
                               'Ratchet', 'Rays and skates', 'Red panda', 'Remote control', 'Scale', 'Scorpion',
                               'Skunk', 'Soap dispenser', 'Spice rack', 'Squid', 'Stapler', 'Stethoscope',
                               'Submarine', 'Syringe', 'Toothbrush', 'Tree house', 'Unicycle', 'Waffle iron',
                               'Wardrobe', 'Whisk', 'Wine rack', 'Worm']]
    not_negative = [d2[f] for f in LEVEL_2_LABELS + LEVEL_3_LABELS + LEVEL_4_LABELS + LEVEL_5_LABELS]

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))

        # Remove Group Of boxes!
        if remove_group_of:
            boxes = boxes[boxes['IsGroupOf'] == 0].copy()
            print(len(boxes))

        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
        print(len(reduced_boxes))

        negative_classes = boxes[boxes['LabelName'].isin(negative_sample_classes)]['ImageID'].unique()
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(set(negative_classes) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-1.csv', index=False)

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))

        # Remove Group Of boxes!
        if remove_group_of:
            boxes = boxes[boxes['IsGroupOf'] == 0].copy()
            print(len(boxes))

        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
        print(len(reduced_boxes))

        negative_classes = boxes[boxes['LabelName'].isin(negative_sample_classes)]['ImageID'].unique()
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(set(negative_classes) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-1.csv', index=False)


def create_level2_files():
    out_dir = OUTPUT_PATH + 'level_2_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    remove_group_of = True

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-2.csv', 'w')
    for l in LEVEL_2_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    parents = get_parents_labels()
    print(parents)
    lvl2_specific_labels = ['Flying disc', 'Heater', 'Hair dryer', 'Humidifier', 'Dishwasher', 'Hand dryer',
                            'Calculator', 'Stapler', 'Pencil sharpener', 'Eraser', 'Fax', 'Pencil case', 'Paper cutter',
                            'Mixing bowl', 'Cocktail shaker', 'Waffle iron', 'Dishwasher', 'Tree house', 'Cantaloupe',
                            'Magpie', 'Isopod', 'Squid', 'Panda', 'Rays and skates', 'Toothbrush', 'Cream', 'Diaper',
                            'Banjo', 'Harmonica', 'Chime', 'Maracas', 'Axe', 'Bomb']
    not_negative = [d2[f] for f in LEVEL_3_LABELS + LEVEL_4_LABELS + LEVEL_5_LABELS]
    list_of_childs = LEVEL_1_LABELS + lvl2_specific_labels

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        # Remove Group Of boxes!
        if remove_group_of:
            reduced_boxes = reduced_boxes[reduced_boxes['IsGroupOf'] == 0].copy()
            print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_2_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))
        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(
            set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-2.csv', index=False)

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        # Remove Group Of boxes!
        if remove_group_of:
            reduced_boxes = reduced_boxes[reduced_boxes['IsGroupOf'] == 0].copy()
            print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_2_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))
        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(
            set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-2.csv', index=False)


def create_level3_files():
    out_dir = OUTPUT_PATH + 'level_3_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-3.csv', 'w')
    for l in LEVEL_3_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    parents = get_parents_labels()
    print(parents)
    lvl3_specific_labels = ['Squid', 'Submarine', 'Panda', 'Red panda']
    not_negative = [d2[f] for f in LEVEL_4_LABELS + LEVEL_5_LABELS]
    list_of_childs = LEVEL_1_LABELS + LEVEL_2_LABELS + lvl3_specific_labels

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_3_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))

        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(
            set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-3.csv', index=False)

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_3_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))

        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(
            set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-3.csv', index=False)


def create_level4_files():
    out_dir = OUTPUT_PATH + 'level_4_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-4.csv', 'w')
    for l in LEVEL_4_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    parents = get_parents_labels()
    print(parents)
    lvl4_specific_labels = ['Unicycle', 'Isopod', 'Squid', 'Scorpion', 'Worm']
    not_negative = [d2[f] for f in LEVEL_5_LABELS]
    list_of_childs = LEVEL_1_LABELS + LEVEL_2_LABELS + LEVEL_3_LABELS + lvl4_specific_labels

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_4_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))

        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-4.csv', index=False)

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_4_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))

        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(
            set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-4.csv', index=False)


def create_level5_files():
    out_dir = OUTPUT_PATH + 'level_5_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-5.csv', 'w')
    for l in LEVEL_5_LABELS:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    parents = get_parents_labels()
    print(parents)
    lvl5_specific_labels = ['Wine rack', 'Wardrobe', 'Closet', 'Unicycle', 'Submarine', 'Magpie', 'Isopod', 'Squid',
                            'Scorpion', 'Worm', 'Mammal', 'Panda', 'Red panda', 'Koala', 'Hippopotamus', 'Hedgehog',
                            'Skunk', 'Armadillo', 'Rays and skates']
    list_of_childs = LEVEL_1_LABELS + LEVEL_2_LABELS + LEVEL_3_LABELS + LEVEL_4_LABELS + lvl5_specific_labels

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_5_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))
        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        negative_samples = list(set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)

        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-5.csv', index=False)

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))
        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)].copy()
        print(len(reduced_boxes))

        parts_list = []
        for lvl1 in list_of_childs:
            for p in parents[lvl1]:
                if p in LEVEL_5_LABELS:
                    print('{} - {} ({})'.format(p, lvl1, d2[lvl1]))
                    small_part = boxes[boxes['LabelName'] == d2[lvl1]].copy()
                    small_part['LabelName'] = d2[p]
                    parts_list.append(small_part)
                    print(len(small_part))
        reduced_boxes = pd.concat([reduced_boxes] + parts_list, axis=0)
        negative_samples = list(set(boxes['ImageID'].unique()) - set(reduced_boxes['ImageID'].unique()))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-5.csv', index=False)


def create_level1_low_samples_files():
    out_dir = OUTPUT_PATH + 'level_1_low_samples_files/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    remove_group_of = True

    labels_to_find = []
    d1, d2 = get_description_for_labels()
    out = open(out_dir + 'class-descriptions-boxable-level-1.csv', 'w')
    for l in LEVEL_1_LABELS_LOW_SAMPLES:
        out.write("{},{}\n".format(d2[l], l))
        labels_to_find.append(d2[l])
    out.close()

    neg_samp_l1 = list(set(LEVEL_1_LABELS) - set(LEVEL_1_LABELS_LOW_SAMPLES))
    print(len(neg_samp_l1))

    negative_sample_classes = [d2[f] for f in ['Armadillo', 'Axe', 'Balance beam', 'Band-aid', 'Banjo', 'Bomb', 'Bottle opener',
                               'Bowling equipment', 'Calculator', 'Can opener', 'Cantaloupe', 'Cassette deck',
                               'Cat furniture', 'Chainsaw', 'Cheese', 'Chime', 'Chisel', 'Closet',
                               'Cocktail shaker', 'Cooking spray', 'Cream', 'Diaper', 'Dishwasher', 'Drill',
                               'Eraser', 'Face powder', 'Facial tissue holder', 'Fax', 'Flying disc', 'Grinder',
                               'Hair dryer', 'Hair spray', 'Hammer', 'Hand dryer', 'Harmonica', 'Heater',
                               'Hedgehog', 'Hiking equipment', 'Hippopotamus', 'Horizontal bar', 'Human body',
                               'Humidifier', 'Indoor rower', 'Ipod', 'Isopod', 'Jacuzzi', 'Koala', 'Ladle',
                               'Lipstick', 'Magpie', 'Maracas', 'Milk', 'Mixing bowl', 'Panda', 'Paper cutter',
                               'Parking meter', 'Pencil case', 'Pencil sharpener', 'Perfume', 'Pizza cutter',
                               'Ratchet', 'Rays and skates', 'Red panda', 'Remote control', 'Scale', 'Scorpion',
                               'Skunk', 'Soap dispenser', 'Spice rack', 'Squid', 'Stapler', 'Stethoscope',
                               'Submarine', 'Syringe', 'Toothbrush', 'Tree house', 'Unicycle', 'Waffle iron',
                               'Wardrobe', 'Whisk', 'Wine rack', 'Worm'] + neg_samp_l1]
    not_negative = [d2[f] for f in LEVEL_2_LABELS + LEVEL_3_LABELS + LEVEL_4_LABELS + LEVEL_5_LABELS]

    if 1:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
        print(len(boxes))

        # Remove Group Of boxes!
        if remove_group_of:
            boxes = boxes[boxes['IsGroupOf'] == 0].copy()
            print(len(boxes))

        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
        print(len(reduced_boxes))

        negative_classes = boxes[boxes['LabelName'].isin(negative_sample_classes)]['ImageID'].unique()
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(set(negative_classes) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'validation-annotations-bbox-level-1.csv', index=False)

    if remove_group_of:
        boxes = pd.read_csv(DATASET_PATH + 'annotations/train-annotations-bbox.csv')
        print(len(boxes))

        # Remove Group Of boxes!
        if 1:
            boxes = boxes[boxes['IsGroupOf'] == 0].copy()
            print(len(boxes))

        reduced_boxes = boxes[boxes['LabelName'].isin(labels_to_find)]
        print(len(reduced_boxes))

        negative_classes = boxes[boxes['LabelName'].isin(negative_sample_classes)]['ImageID'].unique()
        additional_remove = boxes[boxes['LabelName'].isin(not_negative)]['ImageID'].unique()
        print('Additional images to remove: {}'.format(len(additional_remove)))
        negative_samples = list(set(negative_classes) - set(reduced_boxes['ImageID'].unique()) - set(additional_remove))
        print('Length of negative samples: {}'.format(len(negative_samples)))
        neg_samp = get_empty_df(negative_samples)

        reduced_boxes = pd.concat([reduced_boxes, neg_samp], axis=0)
        reduced_boxes.to_csv(out_dir + 'train-annotations-bbox-level-1.csv', index=False)


if __name__ == '__main__':
    create_level1_files()
    if 0:
        create_level2_files()
        create_level3_files()
        create_level4_files()
        create_level5_files()
        create_level1_low_samples_files()
