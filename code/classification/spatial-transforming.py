# spatial-transforming layer
import torch
import torch.nn as nn


class SpatialTransforming(nn.Module):
    """Applies SpatialTransforming as described by Song et. al
    Inspiration for the structure of this code shamelessly ripped from
    the pytorch implementation of the Attention layer.
    Ref: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html 
    """
    def __init__(self, dimensions_in, dimensions_v, dimensions_k):
        super(SpatialTransforming, self).__init__()

        self.linear_key = nn.Linear(dimensions_in, dimensions_k, bias=False)
        self.linear_query = nn.Linear(dimensions_in, dimensions_k, bias=False)
        self.linear_values = nn.Linear(dimensions_in, dimensions_v, bias=False)

        self.linear_out = nn.Linear(dimensions_k * 2, dimensions_k, bais=False) # TODO: what are the proper dimensions
        self.softmax = nn.Softmax(dim=-1)
        # no clipping necessary
    def forward(self, X):
        """
        Args:
        Returns:
        """
        