
# Modelling CIFAR10 - Using Cuper Convergence Strategies

### The requirements in detail:
  
1. Write a code that draws this curve (without the arrows). In submission, you'll upload your drawn curve and code for that

2. Write a code which

   1. uses this new ResNet Architecture for Cifar10:
      - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
      - Layer1 -
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
            R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
            Add(X, R1)
      - Layer 2 -
            Conv 3x3 [256k]
            MaxPooling2D
            BN
            ReLU
      - Layer 3 -
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
            R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
            Add(X, R2)
      - MaxPooling with Kernel Size 4
      - FC Layer 
      - SoftMax
   2. Uses One Cycle Policy such that:
      - Total Epochs = 24
      - Max at Epoch = 5
      - LRMIN = FIND
      - LRMAX = FIND
      - NO Annihilation
   3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
   4. Batch size = 512
   5. Target Accuracy: 90%. 

### Approach Taken:
- Blindly considering that 2000 is a good number of iterations, and also ignoring the suggestion from paper (increase LR linearly - I increased it exponentially), I ran the model with lr ranging from 1e-8 to 1e+2, step 10x
- Because, I had already spent some time in the above, I decided let me see if this approach gave me useful results. The LR range I got from the exercise was from somewhere between 0.001 To 0.1
- I then experimented with 0.1 as max lr and 1/10th 0.01 as min_lr. another round with 0.01 as max and 0.001 as min. Both these ranges got me only until 77%
- I then realized that I had used padding before cutout and a centercrop after that (which was probably not required?). After removing these augmentations around cutout, the model could achieve an accuracy upto 85.5%
- Final step, I thought increasing the lr range a bit (from 0.001 to 0.1) might help and it did. The model reached 90% accuracy in the 24th Epoch


