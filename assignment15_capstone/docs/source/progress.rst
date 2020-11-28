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

  - After creating a dummy dataset with 1 image in the above structure, I ran evaluate and got these results:
    | Segmentation output of planercnn:
    | ..image:: 
