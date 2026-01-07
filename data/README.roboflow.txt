
Bench Press Detection - v5 2023-06-22 4:35pm
==============================

This dataset was exported via roboflow.com on November 5, 2023 at 4:14 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1010 images.
Elbows-shoulders-bar-athlete are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -25 and +25 degrees

The following transformations were applied to the bounding boxes of each image:
* 50% probability of vertical flip
* Random rotation of between -26 and +26 degrees


