import torch
import torch.nn as nn
from modules.TransformerBlock import TransformerBlock
from modules.PositionalEncoding import TimePositionalEncoding
from utils import constants


class TimeModule(nn.Module):

    def __init__(self, emb_module, att_layer, att_heads, dropout, device):
        super(TimeModule, self).__init__()

        self.dropout = dropout
        self.emb_all = emb_module
        self.n_user, self.dim = self.emb_all.weight.size()
        self.device = device

        # Temporal Modules
        self.n_layers = att_layer
        self.stack_att_layers = nn.ModuleList(
            [TransformerBlock(
                input_size=self.dim, d_k=self.dim, d_v=self.dim, n_heads=att_heads, attn_dropout=self.dropout
            ) for _ in range(self.n_layers)])
        self.FC1 = nn.Linear(self.dim, self.dim)
        self.FC2 = nn.Linear(self.dim, self.dim)
        # time positional encoding
        self.layer_temporal_encoding = TimePositionalEncoding(self.dim, device=self.device)
        self.factor_intensity_base = torch.zeros(1, self.n_user, device=self.device, requires_grad=True)
        self.factor_intensity_decay = torch.empty([1, self.n_user], device=self.device, requires_grad=True)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(self.dim, self.n_user)
        self.layer_intensity_hidden.weight.data.fill_(0.01)

        self.time_decoder = nn.Linear(self.dim, self.n_user)

        self.softplus = nn.Softplus()

    def forward(self, src_cas, src_time):

        # src_time = src_time / src_time.max()
        input_time_embeds = self.layer_temporal_encoding(src_time)
        att_hidden = self.emb_all(src_cas)

        # Self-Attention
        mask = (src_cas == constants.PAD)  # build sequence mask
        for att_layer in self.stack_att_layers:
            att_hidden =  input_time_embeds + att_hidden
            att_hidden = att_layer(att_hidden, att_hidden, att_hidden, mask)
        # att_hidden = torch.nn.ReLU()(self.FC2(self.FC1(att_hidden)))

        return att_hidden

    def IntensityCalculation(self, att_hidden, label_times):
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]
        intensity_states = factor_intensity_decay * label_times[:, :, None] + self.layer_intensity_hidden(
            att_hidden) + factor_intensity_base

        return intensity_states
