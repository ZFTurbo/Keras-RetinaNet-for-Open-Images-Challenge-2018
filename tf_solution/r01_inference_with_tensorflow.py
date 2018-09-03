# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_utils_and_constants import *
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# What model to download.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODELS_PATH + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODELS_PATH + 'oid_bbox_trainable_label_map.pbtxt')
NUM_CLASSES = 546

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def run_inference_for_images(images, graph, mirror_images=False):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            o = []
            for i, image in enumerate(images):
                print('Detect: {}'.format(i))
                if mirror_images:
                    image = image[:, ::-1, :]
                image_expanded = np.expand_dims(image, axis=0)
                output_dict = sess.run(tensor_dict,  feed_dict={image_tensor: image_expanded})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint32)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                if mirror_images:
                    tmp = 1 - output_dict['detection_boxes'][:output_dict['num_detections'], 1]
                    output_dict['detection_boxes'][:output_dict['num_detections'], 1] = 1 - output_dict['detection_boxes'][:output_dict['num_detections'], 3]
                    output_dict['detection_boxes'][:output_dict['num_detections'], 3] = tmp
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                o.append(output_dict)

    return o


def run_inference_for_files(files, out_dir, mirror_images):
    batch_size = 1000
    for batch in range(0, len(files), batch_size):
        batch_files = files[batch:batch+batch_size]

        image_arr = []
        scales = []
        ids = []
        for image_path in batch_files:
            id = os.path.basename(image_path)[:-4]
            out_path = out_dir + id + '.pklz'
            if os.path.exists(out_path):
                print('Skip {}'.format(id))
                continue
            print('Read {}'.format(id))
            if 0:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
            else:
                image_np = read_single_image(image_path)
            image_np, scale = resize_image(image_np, min_side=600, max_side=1024)
            image_arr.append(image_np.copy())
            scales.append(scale)
            ids.append(id)

        if len(image_arr) == 0:
            continue

        # Actual detection
        start_time = time.time()
        output_dict = run_inference_for_images(image_arr, detection_graph, mirror_images)
        print('Detection time: {:.3f} sec'.format(time.time() - start_time))

        # Store results
        for i, image_np in enumerate(image_arr):
            out_path = out_dir + ids[i] + '.pklz'
            save_in_file((scales[i], output_dict[i]), out_path)

        if 0:
            # Visualization of the results of a detection.
            for i, image_np in enumerate(image_arr):
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict[i]['detection_boxes'],
                    output_dict[i]['detection_classes'],
                    output_dict[i]['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                show_resized_image(image_np)


def run_inference_kaggle_tst(reverse=False, mirror_images=False):
    files = glob.glob(INPUT_PATH + 'kaggle/challenge2018_test/*.jpg')
    if reverse is True:
        files = files[::-1]

    print(len(files))
    if mirror_images:
        out_dir = OUTPUT_PATH + 'cache_tensorflow_mirror/'
    else:
        out_dir = OUTPUT_PATH + 'cache_tensorflow/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    run_inference_for_files(files, out_dir, mirror_images)


def run_inference_validation(reverse=False, mirror_images=False):
    files = glob.glob(DATASET_PATH + 'validation_big/*.jpg')
    if reverse is True:
        files = files[::-1]
    print(len(files))
    if mirror_images:
        out_dir = OUTPUT_PATH + 'cache_tensorflow_validation_mirror/'
    else:
        out_dir = OUTPUT_PATH + 'cache_tensorflow_validation/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    run_inference_for_files(files, out_dir, mirror_images)


def run_inference_tst(reverse=False, mirror_images=False):
    files = glob.glob(DATASET_PATH + 'test/*.jpg')
    if reverse is True:
        files = files[::-1]
    print(len(files))
    if mirror_images:
        out_dir = OUTPUT_PATH + 'cache_tensorflow_test_mirror/'
    else:
        out_dir = OUTPUT_PATH + 'cache_tensorflow_test/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    run_inference_for_files(files, out_dir, mirror_images)


if __name__ == '__main__':
    run_inference_kaggle_tst(reverse=False, mirror_images=False)
    # run_inference_validation(reverse=False, mirror_images=False)
    # run_inference_tst(reverse=False, mirror_images=False)