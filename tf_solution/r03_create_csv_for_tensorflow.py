# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_utils_and_constants import *
from .r02_validation_with_tesnorflow import get_class_name_mappings


def create_csv_for_tf_predictions(input_dir, out_file):
    out = open(out_file, 'w')
    out.write('ImageId,PredictionString\n')
    tc1, tc2 = get_class_name_mappings()
    files = glob.glob(input_dir + '*.pklz')
    for f in files:
        id = os.path.basename(f)[:-5]
        scale, output_dict = load_from_file(f)
        num_detections = output_dict['num_detections']
        classes = output_dict['detection_classes']
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']

        out.write(id + ',')
        for i in range(num_detections):
            label = tc1[classes[i]]
            xmin = boxes[i][1]
            ymin = boxes[i][0]
            xmax = boxes[i][3]
            ymax = boxes[i][2]

            out.write('{} {} {} {} {} {} '.format(label, scores[i], xmin, ymin, xmax, ymax))
        out.write('\n')


if __name__ == '__main__':
    create_csv_for_tf_predictions(OUTPUT_PATH + 'cache_tensorflow/', SUBM_PATH + 'tf_pretrained_model_kaggle_test.csv')
