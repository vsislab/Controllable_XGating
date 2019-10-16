# Controllable_XGating
The code is not finished yet.
To be continue...

# Controllable Video Captioning with POS Sequence Guidance Based on Gated Fusion Network
## Introduction
In this [paper](https://arxiv.org/abs/1908.10072), we propose to guide the video caption generation with POS information, based on a gated fusion of multiple representations of input videos. We construct a novel gated fusion network, with one cross-gating (CG) block, to effectively encode and fuse different types of representations, *e.g.*, the motion and content features. One POS sequence generator relies on this fused representation to predict the global syntactic structure, which is thereafter leveraged to guide the video captioning generation and control the syntax of the generated sentence. 
This code is a Pytorch implement of this work.

## Dependencies
* Python 2.7
* Pytorch 0.3.1.post3
* Cuda 8.0
* Cudnn 7.0.5

## Prepare
1. Download [Inception_ResNet_V2 features]() of MSRVTT-10K RGB frames and [I3D features]() of MSRVTT-10K optical flows, and put them in `datas` folder.
2. Download [pre-trained models](), and put them in `results` folder.
3. Download the automatic evaluation metrics -- [coco-caption]() and put it in `caption_src`.

## Evaluation
```python
cd caption_src/
sh evaluation.sh
```