import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.lorentz import Lorentz
from manifolds.lorentz_functions import Lorentzian_vector, givens_rotations
from manifolds.hyperboloid import Hyperboloid


class LorentzSelfAttention(nn.Module):

    def __init__(self, c, dimension, dropout=0, device=None, max_position=1000):
        super(LorentzSelfAttention, self).__init__()
        self.c = c
        self.d_emb = dimension
        self.manifold = Lorentz()
        self.hyp_op = Hyperboloid()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dimension)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rotary_position_steps = max_position
        self.device = device

        self.QKV_rotation = nn.Embedding(3, self.d_emb)
        self.QKV_rotation.weight.data = 2 * torch.rand((3, self.d_emb), dtype=torch.float) - 1.0
        self.QKV_rotation.weight.data[:, ::2] = 1.0
        self.QKV_rotation.weight.data[:, 1::2] = 0.0

    def do_rotation(self, x, r, tag=0):
        ori_size = x.size()
        x = x.view(-1, self.d_emb)
        rot_idx = torch.ones(x.size(0), dtype=torch.long).to(x.device)
        rot_idx = rot_idx * tag
        rot_m = F.embedding(rot_idx, r)
        rotated_x = givens_rotations(rot_m, x).reshape(ori_size)
        return rotated_x

    def forward(self, Q, K=None, V=None, mask=None):

        if K is not None and V is not None:
            query, key, value = Q, K, V
        else:
            query, key, value = Q, Q, Q

        query = self.do_rotation(x=query, r=self.QKV_rotation.weight, tag=0)
        key = self.do_rotation(x=key, r=self.QKV_rotation.weight, tag=1)
        value = self.do_rotation(x=value, r=self.QKV_rotation.weight, tag=2)

        batch_size = key.size(0)

        x = torch.zeros_like(query)  # add time axis
        query = torch.cat([x[:, :, 0:1], query], dim=-1)
        key = torch.cat([x[:, :, 0:1], key], dim=-1)
        value = torch.cat([x[:, :, 0:1], value], dim=-1)

        # def to_Lorentz(x):
        #     x_0 = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.c)
        #     x = torch.cat((x_0, x), -1)
        #     return x

        # query = to_Lorentz(query)
        # key = to_Lorentz(key)
        # value = to_Lorentz(value)

        valid_dim = query.size(-1)

        def shape(x):
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, 1, valid_dim)
            return x.transpose(1, 2)

        key = shape(key)
        value = shape(value)
        query = shape(query)
        key_len = key.size(2)
        inf = -1e18
        attn = (2 + 2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, key_len)
            tri_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(pad_mask.device)
            mask = tri_mask + pad_mask
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, inf)

        attn = self.softmax(attn)
        latent_emb = self.manifold.mid_point(value, attn)
        latent_emb = latent_emb.transpose(1, 2).squeeze(2)
        output = latent_emb

        if valid_dim != self.d_emb:
            output = output[:, :, 1:]

        return output
