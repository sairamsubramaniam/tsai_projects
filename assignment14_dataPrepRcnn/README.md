
# Auto Creating Mask and Depth inputs for RCNN training

We use these two amazing repos for creating depth and segmentation mask for existing images, 
which can then be used  as inputs to train Mask-RCNN network:  

- https://github.com/intel-isl/MiDaS
- https://github.com/NVlabs/planercnn

Running code from the first repo on colab was failry simple.
The second repo for plane-rcnn however had some specific requirements:
  - pytorch has to be 0.4.1
  - cuda had to be 8.0
  - gcc 5.0

With the help of Ajith one of our batchmates, many of us could complete these installations on colab  

We also had to modify two script files - evaluate.py and visualize_utils.py in order to avoid
getting too many files (that we will not be using in our training) as output and thereby save some space  
on google colab and google drive

## Link to the dataset for assignment submission:
https://drive.google.com/drive/folders/1CS5Xt2gHY2W25sSAl4KeK7a_bwzPvtTo?usp=sharing

