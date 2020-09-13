
# Modelling CIFAR10 Dataset Using Resnet18

### The requirements in detail:
  
1. Go through this repository: https://github.com/kuangliu pytorch-cifar
2. Extract the ResNet18 model from this repository and add it to your API/repo. 
3. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
4. Your Target is 85% accuracy. No limit on the number of epochs. 5. Use default ResNet18 code (so params are fixed). 
6. Once done finish S8-Assignment-Solution. 


### The solution:
- Using resnet18, the model could achieve 82% easily within 15 epochs
- The model started struggling after that with drop in accuracy and hovering mostly about 83% 
- After 23rd epoch, learning rate was reduced and weight decay of 1e-5 was added. 
- The model quickly jumped to 85.5% with this change in the next epoch itself, but then stayed there for the next 18 epochs

### Issues:
- The model started overfitting from 6th epochs onwards and by the 42nd, it was heavily overfitting with training accuracy reaching >98%
- The model was trained totally for 42 epochs, but most of the epochs didnt help improve the accuracy that much
- We didnt use learning rates optimally to make the training mroe efficient

