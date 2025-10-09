# ViTaDex

This repository contains the implementation of the ViTaDex from the paper:

#### ViTaDex: Vision-Tactile Fusion for 6D Object-in-hand Pose Estimation in Dexterous Anthropomorphic Manipulation

Robot experiment video: [video](https://youtu.be/uyL60vQR2CI)

Download training model: [model](https://drive.google.com/file/d/13FrZcWP7Ic8xJnmI3OBlFyJ6vXQBt-Sp/view?usp=drive_link)


## VTDexM Dataset format
```
VTDexM
   ├─Banana
   │  ├─6D_pose
   │  │  └─0.npy
   │  ├─depth
   │  │  └─0.png
   │  └─rgb
   │  │  └─0.png
   │  └─tactile
   │  │  └─index
   │  │  │  └─0.txt
   │  │  └─middle
   │  │  │  └─0.txt
   │  │  └─palm
   │  │  │  └─0.txt
   │  │  └─pinky
   │  │  │  └─0.txt
   │  │  └─ring
   │  │  │  └─0.txt
   │  │  └─thumb
   │  │  │  └─0.txt
   ├─zobject_model
   │  ├─banana
   │  
   ├─other_files
   ```


## Datasets

VTDexM Dataset download address: [VTDexM Dataset](https://pan.baidu.com/s/1cMhLEsjy4v2Xl66AQE1gbA).
Extraction code: tong


## Model Evaluation

The trained network can be evaluated using the `test_network.py` script.


