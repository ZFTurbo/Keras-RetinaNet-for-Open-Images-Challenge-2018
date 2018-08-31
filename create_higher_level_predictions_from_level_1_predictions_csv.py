# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_utils_and_constants import *
from a01_ensemble_boxes_functions import *


def extend_boxes(boxes, d1, d2, parents, return_only_new=False):
    intersection_thr = 0.75
    print('Initial boxes: {}'.format(boxes.shape))

    # Add all parents boxes
    new_boxes = []
    for i in range(boxes.shape[0]):
        class_name = d1[boxes[i][0]]
        for p in parents[class_name]:
            if p in d2:
                new_boxes.append(np.array([d2[p]] + list(boxes[i][1:])))
    new_boxes = np.array(new_boxes)

    if len(new_boxes) > 0:
        # Filter them with NMS
        unique_labels = np.unique(new_boxes[:, 0])
        # print('Unique parents [{}]: {}'.format(len(unique_labels), [d1[x] for x in unique_labels]))
        keep_boxes = []
        for u in unique_labels:
            part_boxes = new_boxes[new_boxes[:, 0] == u].copy()
            kp = nms_standard(part_boxes[:, 1:].astype(np.float64).copy(), intersection_thr)
            keep_boxes.append(part_boxes[kp].copy())
        merged_boxes = np.concatenate(keep_boxes, axis=0)
    else:
        merged_boxes = new_boxes.copy()
    print('Found parent boxes: {} Reduced with NMS: {}'.format(len(new_boxes), len(merged_boxes)))

    # Concat with older
    if return_only_new is False:
        if len(merged_boxes) > 0:
            new_boxes = np.concatenate((boxes, merged_boxes), axis=0)
        else:
            new_boxes = boxes.copy()
    else:
        new_boxes = merged_boxes.copy()
    print('Total boxes: {}'.format(new_boxes.shape))
    return new_boxes


def flatten_boxes(boxes):
    s = ''
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            s += str(boxes[i, j]) + ' '
    return s


def create_higher_level_classes_from_csv(input_subm, out_file, return_only_new=False):
    d1, d2 = get_description_for_labels_500()
    parents = get_parents_labels()

    subm = pd.read_csv(input_subm)
    ids = subm['ImageId'].values
    preds = subm['PredictionString'].values
    preds_modified = []
    for i in range(len(ids)):
        print('Go for {}'.format(ids[i]))
        id = ids[i]
        if str(preds[i]) == 'nan':
            preds_modified.append('')
            continue
        arr = preds[i].strip().split(' ')
        if len(arr) % 6 != 0:
            print('Some problem here! {}'.format(id))
            exit()
        boxes = []
        for j in range(0, len(arr), 6):
            part = arr[j:j + 6]
            boxes.append(part)
        boxes = np.array(boxes)
        new_boxes = extend_boxes(boxes, d1, d2, parents, return_only_new)
        box_str = flatten_boxes(new_boxes)
        preds_modified.append(box_str)
    subm['PredictionString'] = preds_modified
    subm.to_csv(out_file, index=False)


if __name__ == '__main__':
    create_higher_level_classes_from_csv(SUBM_PATH + 'retinanet_training_level_1.csv',
                                         SUBM_PATH + 'retinanet_level_1_all_levels.csv',
                                         return_only_new=True)
