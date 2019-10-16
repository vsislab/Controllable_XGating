# Controllable Video Captioning with POS Sequence Guidance Based on Gated Fusion Network
## Introduction
In this [paper](https://arxiv.org/abs/1908.10072), we propose to guide the video caption generation with POS information, based on a gated fusion of multiple representations of input videos. We construct a novel gated fusion network, with one cross-gating (CG) block, to effectively encode and fuse different types of representations, *e.g.*, the motion and content features. One POS sequence generator relies on this fused representation to predict the global syntactic structure, which is thereafter leveraged to guide the video captioning generation and control the syntax of the generated sentence. 
This code is a Pytorch implement of this work.
![image](https://github.com/vsislab/Controllable_XGating/blob/imgs/articture.png)

## Dependencies
* Python 2.7
* Pytorch 0.3.1.post3
* Cuda 8.0
* Cudnn 7.0.5

## Prepare
1. Download [Inception_ResNet_V2 features](https://drive.google.com/drive/folders/1_t590bqVTOpRywWlPXttdawiAazkqZkk?usp=sharing) of MSRVTT-10K RGB frames and [I3D features](https://drive.google.com/drive/folders/1-sjrZc5mpo8RRzGNc36l950f5BacPl78?usp=sharing) of MSRVTT-10K optical flows, and put them in `datas` folder.
2. Download [pre-trained models](https://drive.google.com/drive/folders/15LoqMkl_fGQR1UaFxv4zcJgeKWuQo0tQ?usp=sharing), and put them in `results` folder.
3. Download the automatic evaluation metrics -- [coco-caption](https://github.com/tylin/coco-caption), and link it to `caption_src/` as well as `pos_src/`.
```python
ln -s coco-caption/ caption_src/coco-caption
ln -s coco-caption/ pos_src/coco-caption
```
4. Finally, the document structure of the root path should be like this:
![image](https://github.com/vsislab/Controllable_XGating/blob/imgs/tree.png)

## Evaluation
We provide the pre-trained models of "Ours(IR+M)" and "Ours_RL(IR+M)" in paper to reproduce the result reported in paper. Users can change the command in `evaluation.sh` to reproduce "Ours(IR+M)" or "Ours_RL(IR+M)".

Metrics | Ours(IR_M) | Ours_RL(IR+M)
:-: | :-: | :-: 
BLEU@1 | 0.7875 | 0.8175 |
BLEU@2 | 0.6601 | 0.6788 |
BLEU@3 | 0.5339 | 0.5376 |
BLEU@4 | 0.4194 | 0.4128 |
METEOR | 0.2819 | 0.2869 |
ROUGE-L| 0.6161 | 0.6210 |
CIDEr  | 0.4866 | 0.5337 |

```python
cd caption_src/
sh evaluation.sh
```

## Training
Actually, training in this repository is divided into two steps:
1. Train a global pos generator and extract the global postag features.
```python
cd pos_src/
sh run_train.sh
```
After early stopping, extract and store the postag features in `pos_src/globalpos_features/xxx.hdf5`, where `xxx.hdf5` can be customized at [line36 of pos_src/eval_utils.py](https://github.com/vsislab/Controllable_XGating/blob/master/pos_src/eval_utils.py#L36)
```python
sh run_extract_pos.sh
```
Rember to copy the postag features hdf5 into `datas/`.

2. Train the caption model.
```python
cd caption_src/
sh run_train.sh
```

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the use the following BibTeX entry.
```
@article{wang2019controllable,  title={Controllable Video Captioning with POS Sequence Guidance Based on Gated Fusion Network},  author={Wang, Bairui and Ma, Lin and Zhang, Wei and Jiang, Wenhao and Wang, Jingwen and Liu, Wei},  journal={arXiv preprint arXiv:1908.10072},  year={2019}}
```

## Acknowledge
Special thanks to Ruotian Luo, as the code about [Self-critical Sequence Training](http://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html) was inspired by and references to [his repository](https://github.com/ruotianluo/self-critical.pytorch).