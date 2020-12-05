.. EVA5P1 Capstone - Detection, Depth and Segmentation documentation master file, created by
   sphinx-quickstart on Sat Nov 28 11:26:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EVA5P1 Capstone - Detection, Depth and Segmentation
*******************************************************************************

|
|

**NOTE**: I could not complete this assignment. This doc is only a record of my progress in the assignment.

|
|

Background
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


|
|
|


Progress
========

Studying the networks
---------------------

I tried printing the network structure and they look like this:
  - `Planercnn <https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/network_reference/planercnn>`_
  - `Midas <https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/network_reference/midas>`_
  - `YoloV3 <https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/network_reference/yolo_network>`_

Joining planercnn and midas
---------------------------

Next step was to take the first two networks and join them as described in the "approach" section.

To do that, 

  - I first extracted weights from Midas using this `notebook <https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/others/Save_Midas_Weights_as_PIckle.ipynb>`_
  - Added 'Midas Scratch' network `here <https://github.com/sairamsubramaniam/tsai_projects/blob/1dc8d351becb9df3ef716dfbbc6ab46d36c55dfd/assignment15_capstone/planercnn_midas/models/all_in_one.py#L288>`_
  - Studied the :code:`datasets/*` scripts and found that the planercnn code expects 
    the input data structure to be like this:

    | .
    | └── ScanNet
    |     ├── invalid_indices_test.txt
    |     ├── invalid_indices_train.txt
    |     ├── ScanNet
    |     │   └── Tasks
    |     │       └── Benchmark
    |     │           ├── scannetv1_test.txt
    |     │           └── scannetv1_train.txt
    |     ├── scannetv2-labels.combined.tsv
    |     ├── scans
    |     │   └── 0
    |     │       ├── 0.txt
    |     │       ├── annotation
    |     │       │   ├── plane_info.npy
    |     │       │   ├── planes.npy
    |     │       │   └── segmentation
    |     │       │       └── 0.png
    |     │       └── frames
    |     │           ├── color
    |     │           │   └── 0.jpg
    |     │           ├── depth
    |     │           │   └── 0.png
    |     │           └── pose
    |     │               └── 0.txt
    |     └── scene.txt

  - After creating a dummy dataset with 1 image in the above structure, I ran evaluate using default weights:

Input Image:
 .. image:: https://res.cloudinary.com/ss-da/image/upload/v1607166833/0_giwihm.jpg
  :alt: input image
  :target: https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/planercnn_midas/input_data/0.jpg

Segmentation output of planercnn:
 .. image:: https://res.cloudinary.com/ss-da/image/upload/v1607166855/0_segmentation_0_final_nzr8vq.png
  :alt: segmentation output
  :target: https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/planercnn_midas/output_data/0_segmentation_0_final.png
    
Depth output of midas:
 .. image:: https://res.cloudinary.com/ss-da/image/upload/v1607166854/0_midasDepth_0_ncgam1.png
  :alt: midas depth output
  :target: https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment15_capstone/planercnn_midas/output_data/0_midasDepth_0.png


|
|
|


Unsolved Questions
==========================

In planercnn:

* In scannet, `here <https://github.com/NVlabs/planercnn/blob/2698414a44eaa164f5174f7fe3c87dfc4d5dea3b/datasets/scannet_scene.py#L156>`_ what is being done?

* What does the file plane_info.npy contain?

* Plane detection is treated as instance segmentation here, therefore, instances must also have a class_id ? If yes, the dataset we created has planes that do not match with our 4 classes - hardhat, mask, boots and vest. Does that mean we should use the original scannet classes for training ?


|
|
|




.. toctree::
   :maxdepth: 2
   :caption: Contents:



