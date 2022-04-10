# spatial-transforming layer
import torch
import torch.nn as nn


class SpatialTransforming(nn.Module):
    """Applies SpatialTransforming as described by Song et. al
    Inspiration for the structure of this code shamelessly ripped from
    the pytorch implementation of the Attention layer.
    Ref: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html 
    """
    