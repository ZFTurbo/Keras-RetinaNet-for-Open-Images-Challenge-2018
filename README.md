## Keras-RetinaNet for Open Images Challenge 2018

This code was used to get 15th place in Kaggle Google AI Open Images - Object Detection Track competition: 
https://www.kaggle.com/c/google-ai-open-images-object-detection-track/leaderboard 

Repository contains the following:
* Pre-trained models (with ResNet101 and ResNet152 backbones)
* Example code to get predictions with these models for any set of images
* Code to train your own classifier based on Keras-RetinaNet and OID dataset 
* Code to expand predictions for full 500 classes

## Requirements

Python 3.5, Keras 2.2, [Keras-RetinaNet 0.4.1](https://github.com/fizyr/keras-retinanet)

## Pretrained models

There are 2 RetinaNet models based on ResNet101 and ResNet152 for 443 classes (only Level 1). 

| Backbone | Image Size | Model (training) | Model (inference) | Small validation mAP | Full validation mAP |
| --- | --- | --- | --- | --- |  --- |
| ResNet101 | 728 - 1024 |  |  | 0.4896 | 0.377631 |
| ResNet152 | 600 - 800 |  |  | 0.5028 | 0.384009 |

* Model (training) - can be used to resume training or can be used as pretrain for your own classifier
* Model (inference) - can be used to get prediction boxes for arbitrary images

## Inference 

Example can be found here: retinanet_inference_example.py

You need to change files_to_process = glob.glob(DATASET_PATH + 'validation_big/\*.jpg') to your own set of files.
On output you will get "predictions_\*.csv" file with boxes.

Having these predictions you can expand it to all 500 classes using code from create_higher_level_predictions_from_level_1_predictions_csv.py

## Training

For training you need to download OID dataset (~500 GB images): https://storage.googleapis.com/openimages/web/challenge.html

Next fix paths in a00_utils_and_constants.py

Then to train on OID dataset you need to run python files in following order:

* create_files_for_training_by_levels.py
* retinanet_training_level_1/find_image_parameters.py

then
* retinanet_training_level_1/train_oid_level_1_resnet101.py

or 
* retinanet_training_level_1/train_oid_level_1_resnet152.py


## Method description

* https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64633
 