"""
Custom Loss Function

This loss function is binary cross entropy with logit loss
But it also supports masking of weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitLossWithMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, y, mask):
        """
        @param prediction: Model Prediced values
        @param y: Label Values
        @param mask: Mask of input lengths
        """
        return F.binary_cross_entropy_with_logits(prediction, y, weight=mask)
