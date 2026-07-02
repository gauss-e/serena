import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self,dm_in, dm_out, context_length, dropout, qkv_bias=False) -> None:
    super().__init__()
    self.dm_out = dm_out
    self.w_query = nn.Linear(in_features=dm_in, out_features=dm_out, bias=qkv_bias)
    self.w_key = nn.Linear(in_features=dm_in, out_features=dm_out, bias=qkv_bias)
    self.w_value = nn.Linear(in_features=dm_in, out_features=dm_out, bias=qkv_bias)
    self.dropout = nn.Dropout(p=dropout)
    self.register_buffer('mask',
                         torch.triu(torch.ones(context_length, context_length),
                                    diagonal=1))
  def forward(self,x):
    b, num_tokens, d_in = x.shape
    keys = self.w_key(x)
    values = self.w_value(x)
    queries = self.w_query(x)

    attn_scores = queries @ keys.transpose(1, 2)
    attn_scores.masked_fill(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
    )
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1] ** 0.5, dim=-1
    )
    attn_weights = self.dropout(attn_weights)
    context_vector = attn_weights @ values
    return context_vector