# Acheiving 99.4% accuracy in MNIST in a HIGHLY resource constrained system
---

## Constraints:
- Parameters should be less than 10,000
- Accuracy to be achieved in less than 15 epochs


---

# ITERATION 1 - NOTEBOOK [HERE](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/0_base_from_prev_assignment.ipynb)

## Target
- Since I had already done some iterations in the previous [assignment](https://github.com/sairamsubramaniam/tsai_projects/tree/master/assignment2_MNIST), the easiest way to start was to take the previous model, reduce parameters to below 10k and see how the model performed.

## Results:
**Parameters:** 6,246  
**Best Train Accuracy:** 99.37%  
**Best Test Accuracy:** 99.24%  
  
## Analysis:
- The model gave an accuracy of 99.27% (**99.21%** avg in last 5 epochs) with just **6,246** params in **20** epochs  
- When analyzed which images were predicted incorrectly, (to see if we can apply data augmentation), I found the following insights:
  - The horizontal line in 7 makes the model predict 2
  - The top line of 5 missing is creating misprediction
  - A few images with missing pixels is not being predicted properly
  - A few 5 had slight rotation / shear an was predicted as 3
- Currently, I do not know how to address the first three, but the lst one can be accounted for by adding random rotation to the images  
- That would be the next notebook [here](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/1_data_augmentation_added.ipynb).


---

# ITERATION 2 - NOTEBOOK [HERE](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/1_data_augmentation_added.ipynb)

## Target
- The [previous notebook](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/0_base_from_prev_assignment.ipynb) gave an accuracy of 99.21% to 99.27%
- The plan this time is to add Random Rotation while keeping everythig else same

## Results
**Parameters:** 6,246  
**Best Train Accuracy:** 99.18%  
**Best Test Accuracy:** 99.35%  
  
## Analysis
- The augmentation technique has definitely helped as we see the training accuracy has gone up to 99.35%, while the training accuracy has "regularized" to 99.18% (from 99.37% before)  
- Since we do have some leeway in increasing paramters, that would be my next try. The next notebook is [here](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/2_increased_parameters.ipynb).



---

# ITERATION 3 - NOTEBOOK [HERE](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/2_increased_parameters.ipynb)

## Target
- The [previous notebook](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/1_data_augmentation_added.ipynb) gave an accuracy of 99.30% to 99.39%
- The plan this time is to add a few more parameters and see if the model learns better
- Increased params by increasing kernels in the second last layer from 8 to 16

## Results
**Parameters:** 8,142  
**Best Train Accuracy:** 99.21%  
**Best Test Accuracy:** 99.36% [99.5 in 18th epoch!]  
  
## Analysis
- Since the model is reaching 99.5% in its 18th epoch, it certainly has capacity to learn. We will just need to find a way to make it learn faster (within 15 epochs  
- This could be probably be achieved by tweaking the learning rate schedule, which will ne the strategy in the next [notebook](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/3_differing_lr_schedule.ipynb).  



---

# ITERATION 4 - NOTEBOOK [HERE](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/3_differing_lr_schedule.ipynb)

## Target
- The [previous notebook](https://github.com/sairamsubramaniam/tsai_projects/blob/master/assignment3_MNIST/models_tried/2_increased_parameters.ipynb) gave an accuracy of 99.36%, but it also went up to 99.5% in the 18th epoch
- The plan this time is to use an lr rate scheduler and see if it speeds up training to give 99.4% in 10 epochs
- Since, we are slightly crossing the 8k param limit, we will also remove bias and 1 kernel to get it below 8k params

## Results
**Parameters:** 7,978  
**Best Train Accuracy:** 99.44%  
**Best Test Accuracy:** 99.48%  
  
## Analysis
- I tried changing the lr rate schedule to start with 0.1 and change every 5 steps. It gave good accuracy in the first 6 epochs and then accuracy started goind down  
- I therefore tried the above setting for the first 6 epochs and reverted the optimizer for the later epochs and it worked like a charm!  
- Avg test accuracy is 99.4 from epoch 6 to epoch 15  


---

# RECEPTIVE FIELD CALCULATIONS:

Resolution-In | Kernel | Stride | Padding | Resolution-Out | Jump-In | Jump-Out | Receptive Field | 
--- | --- | --- | --- | --- | --- | --- | --- | 
28 | 3 | 1 | 0 | 26 | 1 | 1 | 3 | 
26 | 3 | 1 | 0 | 24 | 1 | 1 | 5 | 
24 | 2 | 2 | 0 | 12 | 1 | 2 | 7 | 
12 | 1 | 1 | 0 | 12 | 2 | 2 | 7 | 
12 | 3 | 1 | 0 | 10 | 2 | 2 | 11 | 
10 | 3 | 1 | 0 | 8 | 2 | 2 | 15 | 
8 | 3 | 1 | 0 | 6 | 2 | 2 | 19 | 
6 | 3 | 1 | 0 | 4 | 2 | 2 | 23 | 
4 | 4 | 1 | 0 | 1 | 2 | 2 | 29 | 
