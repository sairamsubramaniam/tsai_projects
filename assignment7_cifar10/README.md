# Findings from running these networks for 25 epochs:  
  
- with L1 + BN
- with L2 + BN
- with L1 and L2 with BN
- with GBN
- with L1 and L2 with GBN
  
  
### FINDINGS
1. The best accuracy among the 5 models came from Ghost Batch Normalization without any regularization.
2. Adding both L1 & L2 together seems like a bad idea as it gave the lowest accuracies among all the models
3. Among L1 & L2, L2 seems to have given better accuracy, however, the same lambda was used for both. Given that both of them exist at different scales, this certainly was a mistake and probably the reason for L1 not performing better!
4. The model with Ghost Batch Normalization didnt have its training accuracy surpass test accuracy, suggesting the model was not overfitting and therefore didnt need any other regularizers 
5. Compared to plain Batch Normalization, Ghost Batch Normalization makes the model perform better.  
  
  
### TRAIN ACCURACIES
Network  | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Accuracy
---      | ---        | ---                   | ---                  | ---
L1_BN    | 606.3 | 0.96690 | 0.96672 | 0.96773
L2_BN    | 581.8 | 0.98495 | 0.98491 | 0.98565
L1L2_BN  | 612.6 | 0.96322 | 0.96327 | 0.96465
GBN      | 587.4 | 0.99283 | 0.99332 | 0.99388
L1L2_GBN | 607.6 | 0.95510 | 0.95539 | 0.95630
  
  
### TEST ACCURACIES
Network  | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Accuracy
---      | ---        | ---                   | ---                  | ---
L1_BN    | 606.3 | 0.96470 | 0.96628 | 0.9822
L2_BN    | 581.8 | 0.98627 | 0.98540 | 0.9905
L1L2_BN  | 612.6 | 0.93283 | 0.95216 | 0.9643
GBN      | 587.4 | 0.99341 | 0.99300 | 0.9947 
L1L2_GBN | 607.6 | 0.93958 | 0.92470 | 0.9697
  
  
### TEST ACCURACY CHART
![Test Accuracy Chart By Epochs](https://res.cloudinary.com/ss-da/image/upload/v1598799641/tsai_assignments/acc_chart_ta0unn.png)
  
  
# MISCLASSIFIED IMAGES
![misclassifed images with gbn model](https://res.cloudinary.com/ss-da/image/upload/v1598799641/tsai_assignments/misclassified_images_ljwmmb.png)
  
  
  
  
# Code organization  
  
The colab notebook primarily uses my personal google drive as its working directory.  
The folder consistes of these python modules:  
- **raw_models**: Consists of network definitions e.g. with_bn, with_gbn etc as separate files  
- **custom_utils**: Consists of functions that would be useful for training, testing & analyzing.  
  - **helpers.py**: Contains train, test & train_epochs functions
  - **analytics.pyi**: Contains a function to plot misclassified images
  
Two more folders store model outputs and image outputs:  
- **trained_weights**: Stores state_dicts of models  
- **analysis_materials**: stores charts of test loss, accuracy misclassified images etc. that are helpful for analyzing the process


