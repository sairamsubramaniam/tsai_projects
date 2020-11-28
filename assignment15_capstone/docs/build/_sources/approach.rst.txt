Approach
============================

Requirement
-----------

The assignment was to:  

a) Create an encoder-decoder like network that would take in an image and output:
  - Depth image
  - Object bounding boxes
  - Object segmentation

b) Since, it is difficult to train a network from scratch in free resources like colab,  
we were allowed to use pretrained weights from pre-existing networks like:
  - `Yolo <https://github.com/pjreddie/darknet>`_ for object detection
  - `MiDaS <https://github.com/intel-isl/MiDaS>`_ for depth estimation
  - `Planercnn <https://github.com/NVlabs/planercnn>`_ for segmentation

c) This concession was also made because all the above three networks have Resnet as backbone.
It would therefore be easier to have on encoder to take in an image and three decoders to output
the three different things.

Approach
--------

Planercnn was the most difficult of the three networks, because it required lot of system setup:
  - CUDA 8.0
  - GCC 5.0
  - Pytorch 0.4.0

The plan was therefore to study the code of planercnn and integrate the other two networks alongwith
the trained weights into it.
Once done, we could freeze / train appropriate layers to get better accuracy

