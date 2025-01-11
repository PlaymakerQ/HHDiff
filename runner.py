import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import constants
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod
from utils.metrics import calculate_batch_metrics, calculate_MRR, calculate_time_RMSE
from torch.utils.data import DataLoader
from geoopt.optim import RiemannianAdam
import logging

class Runner:

    def __init__(self, config, model, optim=None, edge_list=None, k_list=[1, 5, 10]):
        # basics
        self.PAD = constants.PAD
        self.device = config['device']
        self.epochs = config['epochs']

        self.w_L_soc_E = config['model']['w_L_soc_E']
        self.w_L_dif_E = config['model']['w_L_dif_E']
        self.w_L_user = config['model']['w_L_user']
        self.w_L_time = config['model']['w_L_time']
        self.w_L_hawk = config['model']['w_L_hawk']

        self.model = model
        if optim is None:
            # self.optim = RiemannianAdam(self.model.parameters(), lr=args.lr, stabilize=1)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        else:
            self.optim = optim

        self.loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=constants.PAD)

        # edge_batch_size = max(int(len(edge_list[0]) * config['edge_ratio']), int(len(edge_list[1] * config['edge_ratio'])))
        edge_batch_size = int(max(len(edge_list[0]), len(edge_list[1])) * config['edge_ratio'])
        logging.info(f"edge batch size: {edge_batch_size}")
        self.social_loader = DataLoader(edge_list[0], batch_size=edge_batch_size, shuffle=True)
        self.diffusion_loader = DataLoader(edge_list[1], batch_size=edge_batch_size, shuffle=True)

        self.emb_optim = self.optim

        # metric calculation
        self.current_epoch = 1
        self.k_list = k_list
        self.best_score = 0.0
        self.best_epoch = 0
        self.best_hits = None
        self.best_maps = None
        self.best_mrr = None
        self.valid_RMSE = None

    def train_user_embeds(self):
        total_loss = 0.0
        total_edge = 0
        for batch_edge in self.diffusion_loader:
            self.emb_optim.zero_grad()
            loss = self.model.train_emb(batch_edge) * self.w_L_dif_E
            loss.backward()
            self.emb_optim.step()
            total_loss += loss.item()
            total_edge += len(batch_edge)
        for batch_edge in self.social_loader:
            self.emb_optim.zero_grad()
            loss = self.model.train_emb(batch_edge) * self.w_L_soc_E
            loss.backward()
            self.emb_optim.step()
            total_loss += loss.item()
            total_edge += len(batch_edge)
        torch.cuda.empty_cache()
        return total_loss / total_edge

    def keep_valid_seqs(self, seqs, pad_value=None):
        pad = self.PAD if pad_value is None else pad_value
        valid_idx = (seqs != pad)
        valid_seqs = seqs[valid_idx]
        return valid_seqs

    def batch_data_process(self, batch):
        batch_input_seqs, batch_label_seqs = [], []
        batch_input_times, batch_label_times = [], []
        max_batch_time = 0.0
        for input_seq, label_seq in batch:
            batch_input_seqs.append(torch.LongTensor(input_seq[0]))
            batch_input_times.append(torch.tensor(input_seq[1], dtype=torch.float32))
            batch_label_seqs.append(torch.LongTensor(label_seq[0]))
            batch_label_times.append(torch.tensor(label_seq[1], dtype=torch.float32))
            if max_batch_time < batch_label_times[-1][-1]:
                max_batch_time = batch_label_times[-1][-1]
        batch_input_seqs = pad_sequence(batch_input_seqs, batch_first=True, padding_value=self.PAD)
        batch_label_seqs = pad_sequence(batch_label_seqs, batch_first=True, padding_value=self.PAD)
        batch_input_times = pad_sequence(batch_input_times, batch_first=True, padding_value=max_batch_time)
        batch_label_times = pad_sequence(batch_label_times, batch_first=True, padding_value=max_batch_time)

        return batch_input_seqs, batch_input_times, batch_label_seqs, batch_label_times

    def user_prediction_loss(self, preds, labels, pad_value):
        valid_idx = (labels != pad_value)
        labels = labels[valid_idx].to(preds.device)
        preds = preds[valid_idx]
        loss = self.loss_function(preds, labels)
        return loss * self.w_L_user

    def time_prediction_loss(self, y_true, y_pred):
        loss = (y_true - y_pred.squeeze()) ** 2
        return loss * self.w_L_time

    def train_epoch(self, train_loader):
        self.model.train()
        emb_loss = self.train_user_embeds()
        bar_info = f'train process [{self.current_epoch:>03d}/{self.epochs:>03d}]'
        with tqdm(total=len(train_loader), desc=bar_info) as progress_bar:
            for batch in train_loader:
                src_cas, src_time, tgt_cas, tgt_time = self.batch_data_process(batch)
                pred_user, pred_time, hawkes_loss = self.model(src_cas.to(self.device), src_time.to(self.device),
                                                               tgt_time.to(self.device), mode='train')
                user_loss = self.user_prediction_loss(pred_user, tgt_cas.to(self.device), self.PAD)
                valid_idx = (src_cas != self.PAD).to(self.device)
                true_time = (tgt_time - src_time).to(self.device)
                time_loss = self.time_prediction_loss(true_time[valid_idx], pred_time[valid_idx]).sum()
                if hawkes_loss is None:
                    loss = user_loss + time_loss
                else:
                    hawkes_loss = hawkes_loss * self.w_L_hawk
                    loss = hawkes_loss + user_loss + time_loss
                self.optim.zero_grad()
                loss.backward(retain_graph=True)
                self.optim.step()
                torch.cuda.empty_cache()
                # progress_bar.set_postfix(emb_loss=emb_loss,
                #                          hawkes_loss=hawkes_loss,
                #                          user_loss=user_loss.item(),
                #                          time_loss=time_loss.item())
                progress_bar.update(1)
            return emb_loss

    def valid_epoch(self, valid_loader):
        hits_scores, map_scores, MRR, pred_time = self.test_epoch(valid_loader)
        return hits_scores, map_scores, MRR, pred_time

    def test_epoch(self, test_loader):
        self.model.eval()
        hits_scores = {k: 0.0 for k in self.k_list}
        map_scores = {k: 0.0 for k in self.k_list}
        MRR = 0.0
        n_samples = 0
        time_loss_list = []
        bar_info = f'test process [{self.current_epoch:>03d}/{self.epochs:>03d}]'
        with tqdm(total=len(test_loader), desc=bar_info) as progress_bar:
            for batch in test_loader:
                src_cas, src_time, tgt_cas, tgt_time = self.batch_data_process(batch)
                pred_user, pred_time = self.model(
                    src_cas.to(self.device), src_time.to(self.device),
                    tgt_time.to(self.device), mode='test')
                valid_idx = (tgt_cas!=self.PAD)
                y_truth = tgt_cas[valid_idx]
                pred_user = pred_user.cpu()[valid_idx]
                batch_hits, batch_map = calculate_batch_metrics(y_truth, pred_user, self.k_list)
                MRR += calculate_MRR(y_truth, pred_user)
                valid_idx = (src_cas != self.PAD).to(self.device)
                true_time = (tgt_time - src_time).to(self.device)
                time_loss = self.time_prediction_loss(true_time[valid_idx], pred_time[valid_idx])
                time_loss_list.append(time_loss)
                for k in self.k_list:
                    hits_scores[k] += batch_hits[k]
                n_samples += len(y_truth)
                progress_bar.update(1)

        for k in self.k_list:
            hits_scores[k] = hits_scores[k] / n_samples

        MRR = MRR / n_samples

        RMSE = calculate_time_RMSE(torch.cat(time_loss_list))

        return hits_scores, map_scores, MRR, RMSE

    def check_best_result(self, hits, maps, mrr=None, rmse=None):
        if rmse is not None:
            if (self.valid_RMSE is None) or (self.valid_RMSE > rmse):
                self.valid_RMSE = rmse
        current_score = sum(hits.values())
        if (self.best_hits is None) or (self.best_score < current_score):
            self.best_score = current_score
            self.best_hits = hits
            self.best_maps = maps
            self.best_epoch = self.current_epoch
            self.best_mrr = mrr
            return True
        else:
            return False
