# -*- coding: utf-8 -*-
"""
phase5_training/model.py
=========================
Binary classifier model definition based on MobileNetV2.

Since we are classifying small grayscale crops (128x128) into 
Present (1) or Absent (0), we adapt a pretrained MobileNetV2 by:
- Modifying the first convolution layer to accept 1 channel.
- Replacing the classifier head with a 2-class output.

Usage
-----
    from phase5_training.model import build_model
    model = build_model(pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def build_model(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)

    # Modify first conv to accept 1-channel grayscale instead of 3-channel RGB
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(1, original_conv.out_channels, 
                         kernel_size=original_conv.kernel_size, 
                         stride=original_conv.stride, 
                         padding=original_conv.padding, 
                         bias=original_conv.bias is not None)
    
    if pretrained:
        # Average the weights across the 3 RGB channels to preserve the learned features
        with torch.no_grad():
            new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias[:] = original_conv.bias

    model.features[0][0] = new_conv

    # Replace classifier
    # MobileNetV2's classifier is a Sequential with Dropout and Linear
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model

if __name__ == "__main__":
    # Test model build and shape
    m = build_model()
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    print("Model output shape:", y.shape)  # Expect [2, 2]
