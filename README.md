## Overview
+ `data/` includes config files and train/validation list files
+ `model/` includes related model and module
+ `tool/` includes training and testing scripts
+ `util/` includes data processing, seed initialization

## Usage
### Requirements
python==3.7, torch==1.6, scipy, opencv-python, tensorboardX

### Dataset
Prepare related datasets: Pascal-5<sup>i</sup> ([VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)) and COCO-20<sup>i</sup> ([COCO 2014](https://cocodataset.org/#download))

### Pre-trained models
- Pre-trained backbones and models can be found in [Google Driver](https://drive.google.com/drive/folders/1dEJL_KSkZZ0nIEy6zwqqb93L4zBDvCV-?usp=sharing)
- Download backbones and put the pth files under `initmodel/` folder

### Test and  Train
+ Specify the path of datasets and pre-trained models in the data/config file
+ Use the following command 
  ```
  sh tool/test.sh|train.sh {data} {model} {split_backbone}
  ```

E.g. Test modelwith ResNet50 on the split 0 of PASCAL-5i:
```
sh tool/test.sh pascal asgnet split0_resnet50
```


## References
The code is based on [semseg](https://github.com/hszhao/semseg) and [PFENet](https://github.com/Jia-Research-Lab/PFENet). Thanks for their great work!
