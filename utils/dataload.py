import os
import logging
import random
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import networkx as nx
from utils import constants
from tqdm import tqdm
import scipy.sparse as sp


def split_cascade_in_day(all_cas, all_times, interval):
    new_all_cas, new_all_times = [], []
    all_lens = []
    for time, cas in zip(all_times, all_cas):
        start_idx = 0
        for k in range(1, len(time)):
            if time[k] - time[k - 1] <= interval:
                if k == len(time) - 1:
                    new_time = (time[start_idx:])
                    new_cas = (cas[start_idx:])
                    if len(new_time) > 1:
                        new_all_cas.append(new_cas)
                        new_all_times.append(new_time)
                        all_lens.append(len(new_cas))
            else:
                new_time = (time[start_idx:k])
                new_cas = (cas[start_idx:k])
                if len(new_time) > 1:
                    new_all_cas.append(new_cas)
                    new_all_times.append(new_time)
                    all_lens.append(len(new_cas))
                start_idx = k
    return new_all_cas, new_all_times


class DataProcess:
    def __init__(self, data_path, data_name, args):
        self.data_path = os.path.join(data_path, data_name)
        self.cas_file = os.path.join(self.data_path, 'cascades.txt')
        self.edge_file = os.path.join(self.data_path, 'edges.txt')
        self.is_split = args['is_split']
        self.split_time = args['split_time']
        # time_base = 60.0 * 60 * 24 if data_name != 't' else 60.0
        time_base = 60.0 * 60 * 24
        self.split_interval = time_base * self.split_time
        self.is_sparse = args['is_sparse']
        self.min_len = args['min_len']
        self.max_len = args['max_len']
        self.with_EOS = args['with_EOS']
        self.time_scale = args['time_scale']
        self.u2idx, cascades, times = self.load_cascades(args['train_rate'], args['valid_rate'])
        self.train, self.valid, self.test = cascades
        self.train_time, self.valid_time, self.test_time = times
        self.n_user = len(self.u2idx)
        self.social_edges = None
        self.diffusion_edges = None

    def load_cascades(self, train_rate=0.8, valid_rate=0.1):

        if self.is_sparse:
            u2idx_dict = self.valid_user_counts()
        else:
            u2idx_dict = {'<blank>': constants.PAD, '</s>': constants.EOS}
        current_index = len(u2idx_dict)
        all_cascades, all_times = [], []
        all_length = []
        with open(self.cas_file, 'r') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Loading cascades..."):
                user_in_cas = set()
                cas_list = []
                time_list = []
                chunks = line.strip().strip(',').split(',')
                for record in chunks:
                    record = record.split(' ')
                    record_length = len(record)
                    if record_length < 2:
                        continue
                    if record_length == 3:
                        root, user, timestamp = record
                        user_in_cas.add(root)
                        if self.is_sparse:
                            if root in u2idx_dict.keys():
                                cas_list.append(u2idx_dict[root])
                                time_list.append(float(timestamp))
                        else:
                            if root not in u2idx_dict.keys():
                                u2idx_dict[root] = current_index
                                current_index += 1
                            cas_list.append(u2idx_dict[root])
                            time_list.append(float(timestamp))
                    else:
                        user, timestamp = record
                    if user not in user_in_cas:
                        user_in_cas.add(user)
                        if self.is_sparse:
                            if user in u2idx_dict.keys():
                                cas_list.append(u2idx_dict[user])
                                time_list.append(float(timestamp))
                        else:
                            if user not in u2idx_dict.keys():
                                u2idx_dict[user] = current_index
                                current_index += 1
                            cas_list.append(u2idx_dict[user])
                            time_list.append(float(timestamp))

                if self.min_len < len(cas_list) < self.max_len:
                    all_cascades.append(cas_list)
                    all_times.append(time_list)
                    all_length.append(len(cas_list))

        # sort in time-order
        order = [i[0] for i in sorted(enumerate(all_times), key=lambda x: x[1])]
        all_cascades = [all_cascades[i] for i in order]
        all_times = [all_times[i] for i in order]

        min_time = all_times[0][0]
        all_cascades, all_times = split_cascade_in_day(all_cascades, all_times, self.split_interval)
        all_time_points = []
        for times in all_times:
            all_time_points += times
        all_time_points = [t - min_time for t in all_time_points]

        normalized_term = np.mean(all_time_points) * self.time_scale
        for i in range(len(all_times)):
            start_time = all_times[i][0]
            for s in range(len(all_times[i])):
                all_times[i][s] = (all_times[i][s] - start_time) / normalized_term

        if self.with_EOS:
            for i in range(len(all_cascades)):
                all_cascades[i].append(constants.EOS)
                all_times[i].append(all_times[i][-1] + 1.0)

        train_idx = int(train_rate * len(all_cascades))
        train_cascades = all_cascades[:train_idx]
        train_times = all_times[:train_idx]
        valid_idx = int((train_rate + valid_rate) * len(all_cascades))
        valid_cascades = all_cascades[train_idx:valid_idx]
        valid_times = all_times[train_idx:valid_idx]
        test_cascades = all_cascades[valid_idx:]
        test_times = all_times[valid_idx:]
        cascades = [train_cascades, valid_cascades, test_cascades]
        times = [train_times, valid_times, test_times]
        return u2idx_dict, cascades, times

    def valid_user_counts(self, min_count=5):
        user_counts = {}
        with open(self.cas_file, 'r') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Loading cascades..."):
                user_in_cas = set()
                chunks = line.strip().strip(',').split(',')
                for record in chunks:
                    record = record.split(' ')
                    record_length = len(record)
                    if record_length < 2:
                        continue
                    if record_length == 3:
                        root, user, _ = record
                        user_in_cas.add(root)
                    else:
                        user, timestamp = record
                    if user not in user_in_cas:
                        user_in_cas.add(user)

                for u in user_in_cas:
                    if u not in user_counts.keys():
                        user_counts[u] = 1
                    else:
                        user_counts[u] += 1

        u2idx_dict = {'<blank>': constants.PAD, '</s>': constants.EOS}
        for u in user_counts.keys():
            if user_counts[u] >= min_count:
                u2idx_dict[u] = len(u2idx_dict)

        return u2idx_dict

    def load_adjacency_matrices(self):
        social_adj = self._build_social_adj()
        diffusion_adj = self._build_diffusion_adj()
        adj_list = [social_adj, diffusion_adj]
        return adj_list

    def _build_social_adj(self):
        social_edges = self._load_social_edges()
        social_adj = np.zeros((self.n_user, self.n_user)).astype(int)
        social_adj[social_edges[:, 1], social_edges[:, 0]] = 1
        social_adj[social_edges[:, 0], social_edges[:, 1]] = 1
        np.fill_diagonal(social_adj, 1)
        social_adj = sp.coo_matrix(social_adj, shape=(self.n_user, self.n_user)).tocsr().astype(np.float32)
        social_adj = normalize(social_adj)
        social_adj = sparse_mx_to_torch_sparse_tensor(social_adj)
        return social_adj

    def _load_social_edges(self):
        G = nx.DiGraph()
        with open(self.edge_file, 'r') as file:
            lines = file.readlines()
            for edge in tqdm(lines, desc='load social edges...'):
                if len(edge.strip()) == 0:
                    continue
                src, tgt = edge.strip('\n').split(',')
                if src in self.u2idx.keys() and tgt in self.u2idx.keys():
                    src, tgt = self.u2idx[src], self.u2idx[tgt]
                    G.add_edge(src, tgt)
        edge_sets = list(G.edges)
        edge_sets = torch.tensor(edge_sets, dtype=torch.long)
        return edge_sets

    def _build_diffusion_adj(self):
        diffusion_edges = self._load_diffusion_edges()
        diff_adj = np.zeros((self.n_user, self.n_user)).astype(int)
        diff_adj[diffusion_edges[:, 1], diffusion_edges[:, 0]] = 1
        diff_adj[diffusion_edges[:, 0], diffusion_edges[:, 1]] = 1
        np.fill_diagonal(diff_adj, 1)
        diff_adj = sp.coo_matrix(diff_adj, shape=(self.n_user, self.n_user)).tocsr().astype(np.float32)
        diff_adj = normalize(diff_adj)
        diff_adj = sparse_mx_to_torch_sparse_tensor(diff_adj)
        return diff_adj

    def _load_diffusion_edges(self):
        diffusion_edges = []
        with tqdm(total=len(self.train), desc="load diffusion edges...") as bar:
            for seq in self.train:
                for i in range(len(seq) - 1):
                    if seq[i] != seq[i + 1]:
                        diffusion_edges.append((seq[i], seq[i + 1]))
                bar.update(1)
        return torch.LongTensor(diffusion_edges)

    def load_edge_set(self):
        social_edges = self._load_social_edges()
        diffusion_edges = self._load_diffusion_edges()
        edge_list = [social_edges, diffusion_edges]
        logging.info(f"social edges: {len(social_edges)} and diffusion edges: {len(diffusion_edges)}")
        return edge_list


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor. """
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def normalize(mx):
    """ Row-normalize sparse matrix. """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class CascadeLoader(Dataset):
    def __init__(self, cascades, times, args):

        self.input_seqs = []
        self.label_seqs = []
        for cas, time in zip(cascades, times):
            # assume with_EOS is True
            if len(cas) < args['min_cas']:
                continue
            if len(cas) > args['max_cas']:
                # cas = cas[:args.max_cas]
                cas = cas[-args['max_cas']:]
                time = time[-args['max_cas']:]
            input_seq = cas[:-1]
            input_time = time[:-1]
            label_seq = cas[1:]
            label_time = time[1:]
            self.input_seqs.append([input_seq, input_time])
            self.label_seqs.append([label_seq, label_time])

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs)
        return len(self.input_seqs)

    def __getitem__(self, index):
        return self.input_seqs[index], self.label_seqs[index]
