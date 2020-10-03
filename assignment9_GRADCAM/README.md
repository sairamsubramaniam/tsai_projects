
# Modelling CIFAR10 - Achieve 87% Accuracy and use Albumentations for Image Augmentation

### The requirements in detail:
  
1. Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module. 
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%   

### The solution:
- Using resnet18, the model could achieve 85% easily within 20 epochs
- The model started struggling after that with very high overfitting (train 99.8%)
- After adding augmentations like Rotate and HorizontalFlip, overfitting reduces and gives leeway to continue training
- However with the current strategy, the model could only reach upto 86.99%
- LR scheduling: 0.1 for the first 20 epochs, 0.01 for the next 20 and 0.001 for the last 20

