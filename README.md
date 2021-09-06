# WakeNet
A CNN-based optical image ship wake detector. It uses SWIM as the benchmark dataset.

## Introduction
Most existing wake detection algorithms use **Radon transform (RT)** due to the long streak features of ship wakes in SAR images. The high false alarm rate of RT requires that the algorithms have significant human intervention for image preprocessing. When processing optical images these algorithms are greatly challenged because it is not only sea clutter that interferes with the wake detection in the optical waveband, but many other environmental factors.
To solve the problem, in this repo, we address the automatic ship wake detection task in optical images from the idea of the **CNN** to design an end-to end detector as a novel method. The detector processes all the wake textures clamped by the V-shaped Kelvin arms as the object, and conducts detection via the OBB. The extra regression of wake tip coordinates and Kelvin arm direction can improve the hard wake detection performance while predicting the wake heading. Some special structures and strategies are also adopted according to the wake features.

![Introduction](http://github.com/Lilytopia/WakeNet/tree/mdimg/introduction.png)

## Detection Results on SWIM dataset 
| Method | Baseline | Backbone | mAP (AP) |
| :----- | :------: | :------: | ------------: |
| [R<sup>2</sup>CNN](https://github.com/Xiangyu-CAS/R2CNN.pytorch) | Faster R-CNN | ResNet-101 | 65.24% |
| [R-RetinaNet](https://github.com/ming71/Rotated-RetinaNet) | RetinaNet | ResNet-101 | 72.22% |
| [R-YOLOv3](https://github.com/ming71/yolov3-polygon) | YOLOv3 | DarkNet-53 | 54.87% |
| [BBAVectors](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) | CenterNet | ResNet-101 | 66.19% |
| [*WakeNet (Ours)*](https://github.com/Lilytopia/WakeNet) | RetinaNet | FcaNet-101 | **77.04%** |

![Results](http://github.com/Lilytopia/WakeNet/tree/mdimg/results.png)

## Dependencies
Python 3.6.12, Pytorch 1.7.0, OpenCV-Python 3.4.2, CUDA 11.0

## Getting Started
### Installation
Build modules for calculating OBB overlap and performing Radon transform:
```shell
cd $ROOT/utils
sh make.sh
```

### Prepare the ship wake dataset
WakeNet uses **SWIM** dataset collected by us as the benchmark dataset. The images in SWIM dataset are uniformly cropped into the size of `768 × 768`, with `6,960` images for training, `2,320` images for testing, and `2,320` images for evaluation.
1. Download SWIM dataset from [<u>Kaggle</u>](https://www.kaggle.com/lilitopia/swimship-wake-imagery-mass) into the `$ROOT` directory.
2. Modify the dataset path in `datasets/generate_imageset.py` and then run it.
3. Run `datasets/evaluate/swim2gt.py` to generate ground truth labels for evaluation.
- You may modify the above two files to support other datasets.

### Train model
You can modify the training hyperparameters with the default config file `hyp.py`. Then, run the following command.
```shell
python train.py
```

### Evaluate model
Test the mAP on the SWIM dataset. Note that the VOC07 metric is defaultly used.
```shell
python eval.py
```

### Test model
Generate visualization results.
```shell
python demo.py
```

## Code reference
Thanks to [<u>Rotated-RetinaNet</u>](https://github.com/ming71/Rotated-RetinaNet) for providing a reference in our code writing.

