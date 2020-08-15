# Acheiving 99.4% accuracy in MNIST in a resource constrained system
---

## Constraints:
- Parameters should be less than 20,000
- Accuracy to be achieved in less than 20 epochs
- Fully connected layers must no be used

## How did we achieve it
- We started with a plain network with a minimal network that achieved 98.84% accuracy
- After adding 1x1 convolution just before max pooling, the accuracy increased to 99.09%
- Adding batch normalization after every concolution, increased the accuracy to 99.37%
- All the above network had parameters ~12.5k
- After a bit of experimentation, increasing the parameters to ~17.5k did the trick and achieved 99.42% max accuracy
- However, it was just one epoch, while most of the epochs were just below 99.4%
- We therefore decided to experiment a little more and added another layer to the original layer with ~12.5k params and added a GAP layer at the end. Params decreased to ~8.5k since convolution in the penultimate layer went down from 6x6 kernel to 3x3 kernel
- This gave us 99.35% accuracy. 
- Encouraged by the leeway we had in terms of increasing parameters, we increased kernels in two of the middle layers from 16 to 32 and voila! We got a max accuracy of 99.59% with this model!

## The model that gave the accuracy consists of:
- Contains 8 layers with following number of kernels:
  - Conv 8 -> Conv 8 -> Bottlenck -> Conv 32 -> Conv 32 -> Conv 16 -> Conv 16 -> GAP
  - Bottleneck layer = Maxpool + 1x1 Conv
- Only 3x3 convolution used everywhere (except for the bottleneck & GAP layers
- Except the bottleneck & last Conv layer, every convolution is followed by "RELU" & "Batch Normalization"
- No padding has been used anywhere


## Details of experimentation
### Successful experiments:  
  
Manual Search Path | Model Id | Model Desc | Optimizer | Epochs | Params | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Acc | 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
  -> 0 | 0 | 3-8, 3-8, MP, 3-16, 3-16, 3-16, 6-10 | SGD, lr=0.01, momentum=0.9) | 20 |  | 283.06 | 0.98711 | 0.98674 | 0.9884 | 
0 -> 1 | 1 | 3-8, 3-8, MP, 1x1, 3-16, 3-16, 3-16, 6-10 | SGD, lr=0.01, momentum=0.9) | 20 |  | 289.01 | 0.99024 | 0.99046 | 0.9909 | 
1 -> 2 | 2 | Above network + Batch Norm (after every Relu) | SGD, lr=0.01, momentum=0.9) | 20 | 12442 | 287.77 | 0.99247 | 0.99308 | 0.9937 | 
2 -> 2D | 2D | one more layer + GAP | SGD, lr=0.01, momentum=0.9) | 20 | 8142 | 503.51 | 0.99259 | 0.99308 | 0.9935 | 
2D -> 2Da | 2Da | increased kernels from 16 to 32 in 2 of the middle layers | SGD, lr=0.01, momentum=0.9) | 20 | 18606 | 515.72 | 0.99469 | 0.99466 | 0.9959 |  
  
  ---
### Failed Experiments - 1:  
Manual Search Path | Model Id | Model Desc | Optimizer | Epochs | Params | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Acc | 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
2 -> 2B | 2B | Above network + Drouput after 1st layer (@ 20%) | SGD, lr=0.01, momentum=0.9) | 20 | 12442 | 262.65 | 0.99295 | 0.99314 | 0.9936 | 
2B -> 2Ba | 2Ba | Above network + Drouput after 2 layers (@ 20%) | SGD, lr=0.01, momentum=0.9) | 20 | 12442 | 287.97 | 0.99085 | 0.99098 | 0.9919 | 
2Ba -> 2Bb | 2Bb | Above Network - Dropout 10% instead of 20% | SGD, lr=0.01, momentum=0.9) | 20 | 12442 | 295.9 | 0.99286 | 0.99318 | 0.9937 | 
2Bb -> 2Bc | 2Bc | Above Network + Dropout added to all layers | SGD, lr=0.01, momentum=0.9) | 20 | 12442 | 296.06 | 0.99239 | 0.99278 | 0.9933 |  
  
  ---
### Failed Experiments - 2:  
Manual Search Path | Model Id | Model Desc | Optimizer | Epochs | Params | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Acc | 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
2 -> 2C | 2C | Above Network + increased all 8s to 12 & 16s to 20 | SGD, lr=0.01, momentum=0.9) | 20 | 17902 | 292.13 | 0.99326 | 0.993719 | 0.9942 | 
