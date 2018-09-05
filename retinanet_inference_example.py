# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_utils_and_constants import *
from a01_ensemble_boxes_functions import *


def show_image_debug(id_to_labels, draw, boxes, scores, labels):
    from keras_retinanet.utils.visualization import draw_box, draw_caption
    from keras_retinanet.utils.colors import label_color

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.3:
            break

        color = (0, 255, 0)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(id_to_labels[label], score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    show_image(draw)


def get_retinanet_predictions_for_files(files, out_dir, pretrained_model_path, backbone):
    from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
    from keras_retinanet import models

    show_debug_images = False
    show_mirror_predictions = False

    model = models.load_model(pretrained_model_path, backbone_name=backbone)
    print('Proc {} files...'.format(len(files)))
    for f in files:
        id = os.path.basename(f)[:-4]

        cache_path = out_dir + id + '.pkl'
        if os.path.isfile(cache_path):
           continue

        # try:
        image = read_image_bgr_fast(f)

        if show_debug_images:
            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        if backbone == 'resnet152':
            image, scale = resize_image(image, min_side=600, max_side=800)
        elif backbone == 'resnet101':
            image, scale = resize_image(image, min_side=768, max_side=1024)

        # Add mirror
        image = np.stack((image, image[:, ::-1, :]), axis=0)

        # process image
        start = time.time()
        print('ID: {} Image shape: {} Scale: {}'.format(id, image.shape, scale))
        boxes, scores, labels = model.predict_on_batch(image)
        print('Detections shape: {} {} {}'.format(boxes.shape, scores.shape, labels.shape))
        print("Processing time: {:.2f} sec".format(time.time() - start))

        if show_debug_images:
            if show_mirror_predictions:
                draw = draw[:, ::-1, :]
            boxes_init = boxes.copy()
            boxes_init /= scale

        boxes[:, :, 0] /= image.shape[2]
        boxes[:, :, 2] /= image.shape[2]
        boxes[:, :, 1] /= image.shape[1]
        boxes[:, :, 3] /= image.shape[1]

        if show_debug_images:
            if show_mirror_predictions:
                show_image_debug(LEVEL_1_LABELS, draw.astype(np.uint8), boxes_init[1:], scores[1:], labels[1:])
            else:
                show_image_debug(LEVEL_1_LABELS, draw.astype(np.uint8), boxes_init[:1], scores[:1], labels[:1])

        save_in_file_fast((boxes, scores, labels), cache_path)


def create_csv_for_retinanet(input_dir, out_file, label_arr, skip_box_thr=0.05, intersection_thr=0.55, limit_boxes=300, type='avg'):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')
    d1, d2 = get_description_for_labels()
    files = glob.glob(input_dir + '*.pkl')
    for f in files:
        id = os.path.basename(f)[:-4]
        boxes, scores, labels = load_from_file_fast(f)
        filtered_boxes = filter_boxes(boxes, scores, labels, skip_box_thr)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, intersection_thr, type)
        print(id, len(filtered_boxes[0]), len(filtered_boxes[1]), len(merged_boxes))
        if len(merged_boxes) > limit_boxes:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:limit_boxes]

        out.write(id + ',')
        for i in range(len(merged_boxes)):
            label = int(merged_boxes[i][0])
            score = merged_boxes[i][1]
            b = merged_boxes[i][2:]

            google_name = label_arr[label]
            if '/' not in google_name:
                google_name = d2[google_name]

            xmin = b[0]
            if xmin < 0:
                xmin = 0
            if xmin > 1:
                xmin = 1

            xmax = b[2]
            if xmax < 0:
                xmax = 0
            if xmax > 1:
                xmax = 1

            ymin = b[1]
            if ymin < 0:
                ymin = 0
            if ymin > 1:
                ymin = 1

            ymax = b[3]
            if ymax < 0:
                ymax = 0
            if ymax > 1:
                ymax = 1

            if (xmax < xmin):
                print('X min value larger than max value {}: {} {}'.format(label_arr[label], xmin, xmax))
                continue

            if (ymax < ymin):
                print('Y min value larger than max value {}: {} {}'.format(label_arr[label], ymin, ymax))
                continue

            if abs(xmax - xmin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], xmin, xmax))
                continue

            if abs(ymax - ymin) < 1e-5:
                print('Too small diff for {}: {} and {}'.format(label_arr[label], ymin, ymax))
                continue

            str1 = "{} {:.6f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(google_name, score, xmin, ymin, xmax, ymax)
            out.write(str1)
        out.write('\n')


if __name__ == '__main__':
    skip_box_confidence = 0.01
    iou_thr = 0.55
    limit_boxes_per_image = 300
    type = 'avg'

    # files_to_process = glob.glob(INPUT_PATH + 'kaggle/challenge2018_test/*.jpg')
    files_to_process = glob.glob(DATASET_PATH + 'validation_big/*.jpg')

    if 1:
        backbone = 'resnet101'
        pretrained_model_path = MODELS_PATH + 'retinanet_resnet101_level_1_converted.h5'
        labels_list = LEVEL_1_LABELS

    if 0:
        backbone = 'resnet152'
        pretrained_model_path = MODELS_PATH + 'retinanet_resnet152_level_1_converted.h5'
        labels_list = LEVEL_1_LABELS

    output_cache_directory = OUTPUT_PATH + 'cache_retinanet_level_1_{}/'.format(backbone)
    if not os.path.isdir(output_cache_directory):
        os.mkdir(output_cache_directory)

    get_retinanet_predictions_for_files(files_to_process, output_cache_directory, pretrained_model_path, backbone)
    create_csv_for_retinanet(output_cache_directory,
                             SUBM_PATH + 'predictions_{}_{}_{}.csv'.format(skip_box_confidence, iou_thr, type),
                             labels_list,
                             skip_box_confidence, iou_thr, limit_boxes_per_image, type=type)