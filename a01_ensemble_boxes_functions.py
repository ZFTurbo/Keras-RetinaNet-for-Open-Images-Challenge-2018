# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np


def nms_standard(dets, thresh):
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_boxes(boxes, scores, labels, thr):
    new_boxes = []
    for i in range(boxes.shape[0]):
        box = []
        for j in range(boxes.shape[1]):
            label = labels[i, j].astype(np.int64)
            score = scores[i, j]
            if score < thr:
                break
            # Fix for mirror predictions
            if i == 0:
                b = [int(label), float(score), float(boxes[i, j, 0]), float(boxes[i, j, 1]), float(boxes[i, j, 2]), float(boxes[i, j, 3])]
            else:
                b = [int(label), float(score), 1 - float(boxes[i, j, 2]), float(boxes[i, j, 1]), 1 - float(boxes[i, j, 0]), float(boxes[i, j, 3])]
            box.append(b)
        new_boxes.append(box)
    return new_boxes


def filter_boxes_v2(boxes, scores, labels, thr):
    new_boxes = []
    for t in range(len(boxes)):
        for i in range(len(boxes[t])):
            box = []
            for j in range(boxes[t][i].shape[0]):
                label = labels[t][i][j].astype(np.int64)
                score = scores[t][i][j]
                if score < thr:
                    break
                # Mirror fix !!!
                if i == 0:
                    b = [int(label), float(score), float(boxes[t][i][j, 0]), float(boxes[t][i][j, 1]), float(boxes[t][i][j, 2]), float(boxes[t][i][j, 3])]
                else:
                    b = [int(label), float(score), 1 - float(boxes[t][i][j, 2]), float(boxes[t][i][j, 1]), 1 - float(boxes[t][i][j, 0]), float(boxes[t][i][j, 3])]
                box.append(b)
            # box = np.array(box)
            new_boxes.append(box)
    return new_boxes


def find_matching_box(boxes_list, new_box, match_iou=0.55):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def merge_boxes_weighted(box1, box2, w1, w2, type):
    box = [-1, -1, -1, -1, -1, -1]
    box[0] = box1[0]
    if type == 'avg':
        box[1] = ((w1 * box1[1]) + (w2 * box2[1])) / (w1 + w2)
    elif type == 'max':
        box[1] = max(box1[1], box2[1])
    elif type == 'mul':
        box[1] = np.sqrt(box1[1]*box2[1])
    else:
        exit()
    box[2] = (w1*box1[2] + w2*box2[2]) / (w1 + w2)
    box[3] = (w1*box1[3] + w2*box2[3]) / (w1 + w2)
    box[4] = (w1*box1[4] + w2*box2[4]) / (w1 + w2)
    box[5] = (w1*box1[5] + w2*box2[5]) / (w1 + w2)
    return box


def merge_all_boxes_for_image(boxes, intersection_thr=0.55, type='avg'):

    new_boxes = boxes[0].copy()
    init_weight = 1/len(boxes)
    weights = [init_weight] * len(new_boxes)

    for j in range(1, len(boxes)):
        for k in range(len(boxes[j])):
            index, best_iou = find_matching_box(new_boxes, boxes[j][k], intersection_thr)
            if index != -1:
                new_boxes[index] = merge_boxes_weighted(new_boxes[index], boxes[j][k], weights[index], init_weight, type)
                weights[index] += init_weight
            else:
                new_boxes.append(boxes[j][k])
                weights.append(init_weight)

    for i in range(len(new_boxes)):
        new_boxes[i][1] *= weights[i]
    return np.array(new_boxes)
