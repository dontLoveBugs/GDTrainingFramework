# GDTrainingFramework
This is a general distributed deeplearning training framework based on Pytorch 1.0. You only need to add your model and loss function for a specific task.

# Highlights
- **Distributed & Single GPU** Flexible selection between distributed training with multi gpus and a single gpu.
- **Flexible Visulization Implementation** You can implement any visulization function in "your_model.py" for network comprehensive analysis. 
- **Mixed precision training** Support mixed precision training with NVIDIA apex lib.
- **Sync BN** SUpport Sync BN when training with multi gps.

