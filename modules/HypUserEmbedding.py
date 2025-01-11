import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.lorentz_functions import givens_rotations


class HypUserEmbedding(nn.Module):

    def __init__(self, c, user_emb, rot_matrix, n_neg, device):
        super(HypUserEmbedding, self).__init__()
        self.c = c
        self.data_type = torch.float
        self.n_user = user_emb.weight.size(0)
        self.num_negs = n_neg
        self.device = device
        self.emb = user_emb
        # rotation matrices: List[source, target]
        self.rotation_matrix = rot_matrix
        self.bias_fr = nn.Embedding(self.n_user, 1)
        self.bias_fr.weight.data = torch.zeros((self.n_user, 1), dtype=self.data_type)
        self.bias_to = nn.Embedding(self.n_user, 1)
        self.bias_to.weight.data = torch.zeros((self.n_user, 1), dtype=self.data_type)

    def score(self, frs, tos):
        context_vecs = self.get_fr_embedding(frs)
        target_gold_vecs = self.get_to_embedding(tos)
        dist_score = self.similarity_score(context_vecs, target_gold_vecs)
        bias_frs = self.bias_fr(frs)
        bias_tos = self.bias_to(tos).permute(0, 2, 1)
        score = dist_score + bias_frs + bias_tos

        return score

    def get_neg_samples(self, golds):
        # negative sampling
        neg_samples = torch.randint(self.n_user, (golds.shape[0], self.num_negs)).to(self.device)
        return neg_samples

    def get_fr_embedding(self, frs):
        context_vecs = self.emb(frs)
        dim = context_vecs.size(2)
        # context_frs = torch.zeros(frs.size(), dtype=torch.long).cuda()
        # rel_diag_vecs = self.rotation_matrix(context_frs)
        rel_diag_vecs = self.rotation_matrix[0](frs)
        r, x = rel_diag_vecs.view(-1, dim), context_vecs.view(-1, dim)
        context_rot = givens_rotations(r, x).view(context_vecs.size())

        return context_rot

    def get_to_embedding(self, tos):
        context_vecs = self.emb(tos)
        dim = context_vecs.size(2)
        # context_tos = torch.ones(tos.size(), dtype=torch.long).cuda()
        # rel_diag_vecs = self.rotation_matrix(context_tos)
        rel_diag_vecs = self.rotation_matrix[1](tos)
        r, x = rel_diag_vecs.view(-1, dim), context_vecs.view(-1, dim)
        context_rot = givens_rotations(r, x).view(context_vecs.size())

        return context_rot

    def similarity_score(self, context_vecs, target_vecs):
        x = context_vecs
        y = target_vecs
        x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.c)
        y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + self.c)
        u = torch.cat((x, x2_srt), -1)
        v = torch.cat((y, y2_srt), -1)
        vt = v.permute(0, 2, 1)
        uv = torch.bmm(u, vt)
        result = - 2 * self.c - 2 * uv
        score = result.neg()
        return score

    def forward(self, graph):
        frs = graph[:, 0:1]
        tos = graph[:, 1:2]
        to_negs = self.get_neg_samples(tos)
        positive_score = self.score(frs, tos)
        negative_score = self.score(frs, to_negs)
        positive_loss = F.logsigmoid(positive_score).sum()
        negative_loss = F.logsigmoid(-negative_score).sum()
        batch_loss = - (positive_loss + negative_loss)

        return batch_loss
