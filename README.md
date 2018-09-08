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

There are 3 RetinaNet models based on ResNet50, ResNet101 and ResNet152 for [443 classes](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/a00_utils_and_constants.py#L36) (only Level 1). 

| Backbone | Image Size (px) | Model (training) | Model (inference) | Small validation mAP | Full validation mAP |
| --- | --- | --- | --- | --- |  --- |
| ResNet50 | 728 - 1024 | [533 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.1/retinanet_resnet50_level_1.h5) | [178 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.1/retinanet_resnet50_level_1_converted.h5) | 0.4621 | 0.3520 |
| ResNet101 | 728 - 1024 | [739 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.0/retinanet_resnet101_level_1.h5) | [247 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.0/retinanet_resnet101_level_1_converted.h5) | 0.4896 | 0.3776 |
| ResNet152 | 600 - 800 | [918 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.0/retinanet_resnet152_level_1.h5) | [308 MB](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/releases/download/v1.0/retinanet_resnet152_level_1_converted.h5) | 0.5028 | 0.3840 |

* Model (training) - can be used to resume training or can be used as pretrain for your own classifier
* Model (inference) - can be used to get prediction boxes for arbitrary images

## Inference 

Example can be found here: [retinanet_inference_example.py](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/retinanet_inference_example.py)

You need to change [files_to_process = glob.glob(DATASET_PATH + 'validation_big/\*.jpg')](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/retinanet_inference_example.py#L181) to your own set of files.
On output you will get "predictions_\*.csv" file with boxes.

Having these predictions you can expand it to all 500 classes using code from [create_higher_level_predictions_from_level_1_predictions_csv.py](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/create_higher_level_predictions_from_level_1_predictions_csv.py)

## Training

For training you need to download OID dataset (~500 GB images): https://storage.googleapis.com/openimages/web/challenge.html

Next fix paths in [a00_utils_and_constants.py](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/a00_utils_and_constants.py)

Then to train on OID dataset you need to run python files in following order:

* create_files_for_training_by_levels.py
* retinanet_training_level_1/find_image_parameters.py

then
* retinanet_training_level_1/train_oid_level_1_resnet101.py

or 
* retinanet_training_level_1/train_oid_level_1_resnet152.py


## Ensembles

If you have predictions from several models, for example for ResNet101 and ResNet152 backbones, then you can ensemble boxes with script:
* [ensemble_predictions_with_weighted_method.py](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018/blob/master/ensemble_predictions_with_weighted_method.py)

Proposed method increases the overall performance: 

* ResNet101 mAP 0.3776 + ResNet152 mAP 0.3840 gives in result: mAP 0.4220 

## Method description

* https://www.kaggle.com/c/google-ai-open-images-object-detection-track/discussion/64633
 