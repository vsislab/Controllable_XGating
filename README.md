# Controllable_XGating
The code is not finished yet.
To be continue...

# Controllable Video Captioning with POS Sequence Guidance Based on Gated Fusion Network
## Introduction
In this [paper](https://arxiv.org/abs/1908.10072), we propose to guide the video caption generation with POS information, based on a gated fusion of multiple representations of input videos. We construct a novel gated fusion network, with one cross-gating (CG) block, to effectively encode and fuse different types of representations, *e.g.*, the motion and content features. One POS sequence generator relies on this fused representation to predict the global syntactic structure, which is thereafter leveraged to guide the video captioning generation and control the syntax of the generated sentence.