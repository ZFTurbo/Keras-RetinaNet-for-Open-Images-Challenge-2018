## Simple TensorFlow solution for Open Images Challenge 2018

Download **faster_rcnn_inception_resnet_v2_atrous_coco** model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

then run code one by one: 

* **python r01_inference_with_tensorflow.py** - generate raw files with predictions
* **python r02_validation_with_tesnorflow.py** - find validation score 
* **python r03_create_csv_for_tensorflow.py** - prepare CSV file with answer