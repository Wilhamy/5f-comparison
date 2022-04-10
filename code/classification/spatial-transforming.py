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
        self.dimk = dimensions_k
        self.dimv = dimensions_v
        self.dimIn = dimensions_in

        self.linear_key = nn.Linear(dimensions_in, dimensions_k, bias=False)
        self.linear_query = nn.Linear(dimensions_in, dimensions_k, bias=False)
        self.linear_values = nn.Linear(dimensions_in, dimensions_v, bias=False)

        self.linear_out = nn.Linear(dimensions_k * 2, dimensions_k, bais=False) # TODO: what are the proper dimensions
        self.softmax = nn.Softmax(dim=-1)
        # no clipping necessary

    def forward(self, X):
        """
        Args:
            X (:class:`torch.FloatTensor` [batch size, output length, dimensions]) - Is this the context?
            ---
            Alternative:
            Generate context and queries separately (outside this structure)
            Pass context and queries as args.
        Returns:
            yo momma
        TODO: Handle batch size?
        """
        batch_size, output_len, dimensions = X.size()

        keys = self.linear_key(X)
        query = self.linear_query(X)
        values = self.linear_values(X)

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, X.transpose(1, 2).contiguous()) # TODO: Is X.tranpose correct? Is X the context?
        attention_weights = self.softmax(attention_scores) # TODO: should we divide by \sqrt(dk) as per the paper?
        attention_weights = attention_weights.view(batch_size, output_len, self.dimk) # TODO: is self.dimk correct here?

        mix = torch.bmm(attention_weights, X) # TODO: Is X correct here?

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)

        return output, attention_weights # TODO: could attention_weights be the previous iteration's V?