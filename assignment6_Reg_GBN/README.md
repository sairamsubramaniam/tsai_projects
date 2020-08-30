# Findings from running these networks for 25 epochs:  
  
- with L1 + BN
- with L2 + BN
- with L1 and L2 with BN
- with GBN
- with L1 and L2 with GBN
  
Progression of Difference between Train & Test accuracies  
Which one gave the highest accuracy  
Any visible difference between BN & GBN  
How L1 & L2 affect accuracy  


TRAIN ACCURACIES



TEST ACCURACIES
Network  | Train Time | Last 10 Epoch Avg Acc | Last 5 Epoch Avg Acc | Max Accuracy
---      | ---        | ---                   | ---                  | ---
L1_BN    | 606.3 | 0.96470 | 0.96628 | 0.9822
L2_BN    | 581.8 | 0.98627 | 0.98540 | 0.9905
L1L2_BN  | 612.6 | 0.93283 | 0.95216 | 0.9643
GBN      | 587.4 | 0.99341 | 0.99300 | 0.9947 
L1L2_GBN | 





  
# Code organization  
  
The colab notebook primarily uses my personal google drive as its working directory.  
The folder consistes of these python modules:  
- raw_models: Consists of network definitions e.g. with_bn, with_gbn etc as separate files  
- custom_utils: Consists of functions that would be useful for training, testing & analyzing.  
  - helpers.py: Contains train, test & train_epochs functions
  - analytics.py: Contains a function to plot misclassified images
  
Two more folders store model outputs and image outputs:  
- trained_weights: Stores state_dicts of models  
- analysis_materials: stores charts of test loss, accuracy misclassified images etc. that are helpful for analyzing the process
