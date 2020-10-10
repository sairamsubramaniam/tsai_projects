
# Modelling CIFAR10 - LR Finder, Augmentation and GRADCAM Analysis

### The requirements in detail:
  
1. Pick your last code
2. Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
3. Use this repo: https://github.com/davidtvs/pytorch-lr-finder
    - Move LR Finder code to your modules
    - Implement LR Finder (for SGD, not for ADAM)
    - Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
4. Find best LR to train your model
5. Use SDG with Momentum
6. Train for 50 Epochs. 
7. Show Training and Test Accuracy curves
8. Target 88% Accuracy.
9. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
10. Submit


### Observations:
- LR Finder is helpful, but manual experiments of initial learning rates worked better
- For cifar10 with resnet, achieving 80% happens very quickly, but it gets difficult to improve accuracy after that
- However, there was a surprise jump from 81% to 89% in the 20th epoch, after which it stayed between 90 to 91% for a long time
- Augmentations like cutout, rotate, horizontal-flip and normalize helped the model dela overfitting until the 10th epoch, which lead to better performance


