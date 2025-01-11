import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.constants as constants
from modules.HypUserEmbedding import HypUserEmbedding
from modules.RotaryLorentzAttention import LorentzSelfAttention
from utils.others import get_previous_user_mask
from manifolds.hyperboloid import Hyperboloid
from manifolds.lorentz_functions import givens_rotations, distance_prediction_scores
from modules.TimeModule import TimeModule
from utils.run_utils import make_type_mask_for_pad_sequence, compute_loglikelihood


class HHDiff(nn.Module):
    def __init__(self, args):
        super(HHDiff, self).__init__()
        # parameters
        self.device = args['device']
        self.n_user = args['n_user']

        config = args['model']

        self.c = config['c']
        self.dim = config['n_dim']
        self.init_size = config['init_emb_size']
        self.rotary_position_steps = config['rot_pe_steps']
        self.dropout = config['dropout']

        # Computing log-likelihood for the Hawkes prcoess
        self.loss_integral_num_sample_per_step = config['n_sampled_times']

        # hyperbolic
        self.manifold = Hyperboloid()

        # modules
        self.emb_all = nn.Embedding(self.n_user, self.dim, padding_idx=0)  # user embeddings
        init_weight = self.init_size * torch.randn((self.n_user, self.dim))
        # init_weight = self.manifold.expmap0(self.manifold.proj_tan0(init_weight, self.c), self.c)
        self.emb_all.weight.data = init_weight

        self.rot_src = nn.Embedding(self.n_user, self.dim)  # user rotation matrices
        self.rot_src.weight.data[:, ::2] = 1.0
        self.rot_src.weight.data[:, 1::2] = 0.0

        self.rot_tgt = nn.Embedding(self.n_user, self.dim)
        self.rot_tgt.weight.data[:, ::2] = 1.0
        self.rot_tgt.weight.data[:, 1::2] = 0.0

        self.hyp_emb_module = HypUserEmbedding(
            c=self.c, user_emb=self.emb_all, rot_matrix=[self.rot_src, self.rot_tgt],
            n_neg=config['n_neg'], device=self.device)

        # Attention Module
        self.hyp_att = LorentzSelfAttention(c=self.c, dimension=self.dim)
        self.time_att = TimeModule(emb_module=self.emb_all, att_layer=config['n_time_att_layer'],
                                   att_heads=config['n_time_att_heads'], dropout=config['time_drop_out'],
                                   device=self.device)

        # rotary positional encoding
        self.pos_encode = torch.randn(self.rotary_position_steps, self.dim, device=self.device)
        self._set_angles(step=self.rotary_position_steps)
        self.dropout_layer = nn.Dropout(self.dropout)  # dropout layer

        # Hawkes process
        self.influence_intensity_hidden = nn.Linear(self.dim, self.n_user)
        self.temporal_intensity_hidden = nn.Linear(self.dim, self.n_user)
        self.factor_intensity_decay = torch.empty([1, self.n_user], device=self.device)
        nn.init.xavier_normal_(self.factor_intensity_decay)
        self.softplus = nn.Softplus()

        # predictions

        # CAT MANNER
        # self.decode_user = nn.Linear(2*self.dim, self.n_user)

        # PLUST MANNER
        self.decode_user = nn.Linear(self.dim, self.n_user)
        self.decode_time_user = nn.Linear(self.dim, self.n_user)

        # time prediction
        self.decode_time = nn.Linear(self.dim, 1)

        # score weights
        self.w_A_user = config['w_A_user']

        # weight initialization
        # self.decode_user.weight.data.fill_(0.01)
        # torch.nn.init.xavier_normal_(self.decode_user.weight)
        # torch.nn.init.xavier_uniform_(self.decode_user.weight)

    def UserPrediction(self, hid_embeds, time_hiddens):

        # dis_reg = distance_prediction_scores(hid_embeds, self.emb_all.weight)
        pred_user_1 = self.decode_user(hid_embeds)
        pred_user_2 = self.decode_time_user(time_hiddens)
        pred_user = self.w_A_user * pred_user_1 + (1 - self.w_A_user) * pred_user_2

        # pred_user = dis_reg + pred_user

        # a = 0.95
        # pred_user = a * pred_user_1 + (1-a) * pred_user_2
        # pred_user = self.decode_user(hid_embeds + time_hiddens)

        # hiddens = torch.cat([hid_embeds, time_hiddens], dim=-1)
        # pred_user = self.decode_user(hiddens)

        return pred_user

    def TimePrediction(self, hid_embeds):
        pred_time = self.decode_time(hid_embeds)
        return pred_time

    def forward(self, src_cas, src_time, tgt_time, mode='train'):
        # data process
        delta_times = tgt_time - src_time
        delta_times = delta_times / (tgt_time + 1e-6)

        # masks
        att_mask = (src_cas == constants.PAD)
        pred_mask = get_previous_user_mask(src_cas, self.n_user)

        # map users in cascades to hyperbolic user representations
        user_embeds_in_cas = self.emb_all(src_cas)

        # calculate hidden representations (influence, temporal)
        pos_user_embeds_in_cas = self.RotaryPositionalEncoding(user_embeds_in_cas)
        inf_hiddens = self.hyp_att(pos_user_embeds_in_cas, pos_user_embeds_in_cas, pos_user_embeds_in_cas, att_mask)
        time_hiddens = self.time_att(src_cas, src_time)

        # next user prediction
        pred_user = self.UserPrediction(inf_hiddens, time_hiddens)
        pred_user = pred_user + pred_mask

        # next time prediction
        pred_time = self.TimePrediction(inf_hiddens)

        if mode == 'train':
            # use Hawkes process to fit observed historical cascades in training phrase
            valid_mask = ~att_mask
            type_mask = make_type_mask_for_pad_sequence(src_cas, self.n_user)
            hawkes_loss = self.LogLikelihoodLoss(inf_states=inf_hiddens, time_states=time_hiddens,
                                                 time_delta_seqs=delta_times, pad_mask=valid_mask, type_mask=type_mask)
            # hawkes_loss = None
            return pred_user, pred_time, hawkes_loss

        else:
            return pred_user, pred_time

    def LogLikelihoodLoss(self, inf_states, time_states, time_delta_seqs, pad_mask, type_mask):
        """ calculate the loss for cascade fitting via the hyperbolic hawkes process"""

        # prefer_states = 10 / prefer_states

        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        intensity_states = factor_intensity_decay * time_delta_seqs[..., None] + self.temporal_intensity_hidden(
            time_states) + self.influence_intensity_hidden(inf_states)
        lambda_at_event = self.softplus(intensity_states)
        # MCMC to estimate the Lambda
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs)
        state_t_sample = self.compute_states_at_sample_times(
            inf_states=inf_states, event_states=time_states, sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        event_ll, non_event_ll, _ = compute_loglikelihood(
            lambda_at_event=lambda_at_event, lambdas_loss_samples=lambda_t_sample, time_delta_seq=time_delta_seqs,
            seq_mask=pad_mask, lambda_type_mask=type_mask)
        loss = - (event_ll - non_event_ll).sum()
        return loss

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval. """
        dtimes_ratio_sampled = torch.linspace(start=0.0, end=1.0, steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]  # [1, 1, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled
        return sampled_dtimes  # [batch_size, max_len, n_samples]

    def compute_states_at_sample_times(self, inf_states, event_states, sample_dtimes):
        """Compute the hidden states at sampled times. """
        event_states = event_states[:, :, None, :]  # [batch_size, seq_len, 1, hidden_size]
        inf_states = inf_states[:, :, None, :]
        sample_dtimes = sample_dtimes[..., None]  # [batch_size, seq_len, num_samples, 1]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]  # [1, 1, 1, num_event_types]
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.temporal_intensity_hidden(
            event_states) + self.influence_intensity_hidden(inf_states)
        return intensity_states

    def train_emb(self, graph):
        graph = graph.to(self.device)
        emb_loss = self.hyp_emb_module(graph)
        return emb_loss

    def RotaryPositionalEncoding(self, x):
        B, L, dim = x.size()
        seq_idx = torch.arange(L).expand(B, L).to(x.device)
        pos_rot_mat = F.embedding(seq_idx, self.pos_encode)
        rot_pos_emb, x = pos_rot_mat.view(-1, self.dim), x.view(-1, self.dim)
        pos_user_emb = givens_rotations(rot_pos_emb, x).reshape(B, L, -1)
        return pos_user_emb

    def _set_angles(self, step=20):
        num = self.pos_encode.size(0)
        repeat = int(num / step)
        base_angle = math.pi / step
        angles = torch.arange(step) * base_angle
        angles = angles.unsqueeze(0).expand(repeat, step).reshape(num, 1)
        msin = torch.sin(angles)
        mcos = torch.cos(angles)
        self.pos_encode[:, ::2] = mcos
        self.pos_encode[:, 1::2] = msin
