
# YOLO - OBJECT DETECTION


This assignment consists of 3 sections:

## Section 1
### How to perform object detection using opencv's yolo implementation
The notebook and the input / output images for the same are in this folder yolo_opencv
Credit: https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
& TSAI


## Section 2
### Image collection and annotation for YoloV3 training
- In this section, we collected images that contains the following classes: hardhat, vest, mask, boots
- For annotating with bounding boxes, we used this tool: https://github.com/miki998/YoloV3_Annotation_Tool
(which in turn credits, this repo: https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-)
- All images annotated by different students of the TSAI EVA bath were then collatted & given back to us for training


## Section 3
### How to train Yolo in google colab and make video out of images with object detection
- We followed steps given here (https://github.com/theschoolofai/YoloV3)
(which was picked from this repo: https://github.com/ultralytics/yolov3)
to complete the training of bounding boxes in images received from section 2
- For inference, we picked a random video from youtube (I picked this: https://www.youtube.com/watch?v=nDCdRF4FcJE)
- Split it into images using ffmpeg
- Ran yolov3 trained model on it
- And put the images back together into a video uploaded here: https://youtu.be/kJNbsQNP2Iw
