# AIVisionStreamAnalytics 

This video analytics pipeline developed in C++ on top of NVIDIA DeepStream. It enables real-time, pre-recorded processing of video streams using deep learning models, YOLO-based object detection networks. The code is implemented for scalable, high-throughput inference with efficient GPU acceleration (TritonRT - engine file), making it suitable for edge deployments and intelligent video applications. Development and testing have been performed on NVIDIA Jetson Orin Super.

This code support two pipeline.

1. Analyis live video stream
2. Analyis pre-recorded .mp4 video files 

## Live stream Pipeline

```
nvarguscamerasrc -> capsfilter -> nvstreammux -> nvinfer -> nvtracker -> nvdsanalytics -> nvvideoconvert -> nveglglessink (display sink)
```

## Pre-recorded video file

```
filesrc -> qtdemux -> h265parse -> nvv4l2decoder -> nvvideoconvert -> capsfilter -> nvstreammux -> nvinfer -> nvtracker -> nvdsanalytics -> nvvideoconvert -> fakesink (headless)
```

## Config file
- config_analytics -> Deepstream analytic config
- config_infre_primary_yolo -> Primary GIE config
- config_tracker -> tracking config

## mdls folder
- contain onnx yolov8 
- Engine file 
This folder conteint ignored for github

## Usage
``
$cd src
$bash compile.sh
``

![tracker](images/obj_tracker.gif)

- The center box defines the ROI for object counting
- Directional arrows indicate the valid movement direction—only objects moving along this path are analyzed
- A virtual line counts only those objects that cross it in the specified direction (top-to-bottom)