import torch
import numpy as np


def calculate_batch_metrics(y_true, y_pred, k_list):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    hits_scores = {k: 0.0 for k in k_list}
    map_scores = {k: 0.0 for k in k_list}
    max_k = max(k_list)
    top_k_rec = np.argsort(y_pred, axis=-1)[:, -max_k:]
    top_k_rec = np.flip(top_k_rec, axis=1)
    for k in k_list:
        preds_k = top_k_rec[:, :k]
        for top_k, label in zip(preds_k, y_true):
            idx = np.where(top_k == label)[0]
            if len(idx) != 0:
                hits_scores[k] += 1
                map_scores[k] += 1 / (idx[0] + 1)
            else:
                hits_scores[k] += 0
                map_scores[k] += 0
    return hits_scores, map_scores


def calculate_MRR(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    MRR = 0.0
    top_k_rec = np.argsort(y_pred, axis=-1)
    top_k_rec = np.flip(top_k_rec, axis=1)
    for top_k, label in zip(top_k_rec, y_true):
        idx = np.where(top_k == label)[0][0]
        MRR += 1 / (idx + 1)
    return MRR


def calculate_time_RMSE(difference):
    if isinstance(difference, torch.Tensor):
        difference = difference.detach().cpu().numpy()
    RMSE = np.sqrt(np.mean(difference ** 2))

    return RMSE
