# TRAINING CIFAR10 USING DILATED & DEPTHWISE SEPARABLE CONVOLUTIONS
  
### In more detail, the conditions to be achieved for this assignment are:
  
- change the code such that it uses GPU
- change the architecture to C1C2C3C40 (basically 3 MPs)
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
- upload to Github
- Attempt S7-Assignment Solution
  
  
### The network satisfying the above conditions are in raw_models.S7 module  
- 80% was achieved by the 8th Epoch. 
- The model has 777,546 trainable params. 
- Receptive Fields calculations are given below:

Resolution-In | Kernel | Stride | Padding | Resolution-Out | Jump-In | Jump-Out | Receptive Field | 
--- | --- | --- | --- | --- | --- | --- | --- | 
32 | 3 | 1 | 1 | 32 | 1 | 1 | 3 | 
32 | 3 | 1 | 1 | 32 | 1 | 1 | 5 | 
32 | 2 | 2 | 0 | 16 | 1 | 2 | 7 | 
16 | 1 | 1 | 0 | 16 | 2 | 2 | 7 | 
16 | 3 | 1 | 1 | 16 | 2 | 2 | 11 | 
16 | 5 | 1 | 2 | 16 | 2 | 2 | 19 | 
16 | 2 | 2 | 0 | 8 | 2 | 4 | 23 | 
8 | 1 | 1 | 0 | 8 | 4 | 4 | 23 | 
8 | 3 | 1 | 1 | 8 | 4 | 4 | 31 | 
8 | 3 | 1 | 1 | 8 | 4 | 4 | 39 | 
8 | 8 | 1 | 0 | 1 | 4 | 4 | 67 | 


