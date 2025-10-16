<div align="center">
<h3>Multi-Level Feature Fusion Network for Shadow Removal Detection </h3>
</div>

## Overview

In this paper, we present a novel model called Multi-level Feature Fusion Network (MFF-Net) for shadow removal detection. MFF-Net consists of two parts: a dual-branch feature extraction encoder and a dense prediction decoder. 
The encoder anchors the approximate position of the manipulated regions, while the decoder progressively fills in the details of the estimated shadow masks by integrating multi-level information. 
In the encoder part, a global modeling branch is constructed to capture long-range dependencies, while a local feature extraction branch is designed to extract local structural information.
The features extracted by these two branches are integrated using a feature fusion module. 
In the decoder part, a multi-scale feature upsampling module is proposed to upsample the input features and integrate them with the low-level features obtained from the encoder part. 
Meanwhile, the cross attention mechanism is introduced to guide the multi-level feature fusion process. Finally, the features of different resolutions are employed to estimate the shadow masks in a coarse-to-fine manner.

## Contributions

• A novel model called Multi-level Feature Fusion Network (MFF-Net) is proposed for shadow removal detection, and it consists of a dual-branch feature extraction encoder and a dense prediction decoder. The encoder is designed
to anchor the approximate position of the manipulated regions, while the decoder progressively refines the details of the estimated shadow masks by utilizing low-level information.

• We explore and demonstrate the effectiveness of establishing two transformer-based models with different information processing capabilities as parallel backbones for detection tasks. These two branches provide the model with sufficient global modeling capability and local feature extraction capability, respectively. Meanwhile, since both of the two branches are all transformer-based models (even with distinct information processing capabilities), the features extracted from the two branches can be effectively fused to achieve improved performance.

• A novel module called MSFU is proposed to guide the upsampling and fusion process. The MSFU module effectively preserves the internal relationships of features and maintains their spatial structure during the upsampling process. Then, it facilitates a more comprehensive integration of multi-level features and enhances the local structural information for the fused features.
