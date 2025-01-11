import os
import datetime
import torch
import numpy as np
import json
import logging
import yaml


def save_model(state, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, 'model.pt')
    torch.save(state, save_dir)


def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq


def find_target_path(target_folder_name):
    current_path = os.path.abspath(__file__)
    while True:
        current_path, folder_name = os.path.split(current_path)
        if folder_name == target_folder_name:
            return os.path.join(current_path, folder_name)
        if not folder_name:  # 到达根目录
            raise ValueError(f"cannot find folder named {target_folder_name}")

def set_save_path(model_name="no-name", data_name='test', save_name='save', TEST_MODE=False):
    root_path = find_target_path('HHDiff')
    save_root_path = os.path.join(root_path, save_name)
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    save_path = os.path.join(root_path, save_name, data_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # change <folder_path> to set the path for the project.
    if TEST_MODE:
        folder_name = 'TEST'
    else:
        dt = datetime.datetime.now()
        date_str, time_str = dt.strftime("%m%d_"), dt.strftime('%H%M%S_')
        folder_name = date_str + time_str + model_name
    folder_path = os.path.join(save_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path

def set_logging(log_dir, model_name=None, log_name=None):
    """ set log file path and logging formats"""
    formatter = logging.Formatter(
        f"%(asctime)s | {model_name} | %(levelname)-4s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.getLogger("").setLevel(logging.INFO)
    # save in log file
    if log_dir is None:
        filename = os.path.join(log_dir, 'train.log')
    else:
        filename = os.path.join(log_dir, f"{log_name}.log")
    file = logging.FileHandler(filename)
    file.setFormatter(formatter)
    logging.getLogger("").addHandler(file)
    logging.info(f"logging settings finished...")


def load_configs(data_name, configs_path='configs'):
    root_path = find_target_path('HHDiff')
    config_file = os.path.join(root_path, configs_path, f"{data_name}.yaml")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def info_check(config, data_name):
    logging.info(f"base information:")
    logging.info(f" - dataset: {data_name}")
    logging.info(f" - device: {config['device']}")
    logging.info(f" - seed: {config['seed']}")
    logging.info(f" - batch size: {config['batch_size']}")
    logging.info(f" - learning rate: {config['lr']}")
    logging.info(f" - edge learn ratio: {config['edge_ratio']}")
    logging.info(f"data information")
    logging.info(f" - initial minimum length : {config['min_len']}")
    logging.info(f" - time scale             : {config['time_scale']}")
    logging.info(f" - is sparse?             : {config['is_sparse']}")
    logging.info(f" - is split?              : {config['is_split']}")
    if config['is_split']:
        logging.info(f" - split time threshold   : {config['split_time']}")
    logging.info(f" - minimum cascade length : {config['min_cas']}")
    logging.info(f" - maximum cascade length : {config['max_cas']}")
    logging.info(f"model information:")
    for key in config['model'].keys():
        logging.info(f" - {key.ljust(16)} : {config['model'][key]}")

def format_metrics(scores, mrr=None, rmse=None, fix_len=6):
    metric_str = " "
    score_str = " "
    for k in scores.keys():
        metric = f"Hit@{k:02d}"
        score = f"{scores[k]:.4f}"
        metric_str += metric.ljust(fix_len) + " | "
        score_str += score.ljust(fix_len) + " | "
    if mrr is not None:
        metric = "MRR"
        metric_str += metric.ljust(fix_len) + " | "
        score = f"{mrr:.4f}"
        score_str += score.ljust(fix_len) + " | "
    if rmse is not None:
        metric = "RMSE"
        metric_str += metric.ljust(fix_len)
        score = f"{rmse:.3f}"
        score_str += score.ljust(fix_len)
    total_str = metric_str + "\n" + score_str
    return total_str


def log_results(save_file, metric_str, epoch):
    with open(save_file, 'a') as file:
        file.write(f"update best score in epoch {epoch}:\n")
        file.write(metric_str + "\n")


if __name__ == '__main__':
    print("done.")
