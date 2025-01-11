import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import logging
import time
from tqdm import tqdm
from utils.dataload import DataProcess, CascadeLoader
from arguments import parse_arguments
from runner import Runner
from utils.others import load_configs, set_save_path, format_metrics
from utils.others import set_logging, log_results, info_check
from model import HHDiff as Model


def set_seed(seed, CUDA_MODE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_MODE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_arguments()
    config = load_configs(args.data_name)
    PRINT = logging.info

    model_name = Model.__name__
    args.model_name = model_name
    args.save_path = set_save_path(model_name, args.data_name, TEST_MODE=args.TEST_MODE)
    set_logging(args.save_path, args.model_name)
    args.save_file = os.path.join(args.save_path, f'result.txt')
    PRINT(f"save path: {args.save_path}")

    if args.cuda and torch.cuda.is_available():
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')

    set_seed(config['seed'], args.cuda)

    # all default settings and arguments are initialized.
    Data = DataProcess('data', args.data_name, config)
    args.u2idx_dict = Data.u2idx
    config['n_user'] = len(args.u2idx_dict)
    train_cascades = CascadeLoader(Data.train, Data.train_time, config)
    val_cascades = CascadeLoader(Data.valid, Data.valid_time, config)
    test_cascades = CascadeLoader(Data.test, Data.test_time, config)
    train_loader = DataLoader(train_cascades, batch_size=config['batch_size'], shuffle=True, drop_last=False,
                              num_workers=config['workers'], collate_fn=lambda x: x)
    val_loader = DataLoader(val_cascades, batch_size=config['batch_size'], shuffle=False, drop_last=False,
                            num_workers=config['workers'], collate_fn=lambda x: x)
    test_loader = DataLoader(test_cascades, batch_size=config['batch_size'], shuffle=False, drop_last=False,
                             num_workers=config['workers'], collate_fn=lambda x: x)

    edge_list = Data.load_edge_set()
    adj_list = None
    total_train, total_val, total_test = len(train_cascades), len(val_cascades), len(test_cascades)
    PRINT(f"number of users: {config['n_user']}")

    # data loaded
    model = Model(config).to(config['device'])
    run = Runner(config, model, edge_list=edge_list)
    PRINT(f"all preparation is complete...")

    info_check(config, args.data_name)

    PRINT("start training process...")
    early_stopping_check = 0
    for epoch in range(config['epochs']):

        loss = run.train_epoch(train_loader)

        hits_k, map_k, mrr, rmse = run.test_epoch(test_loader)
        metric_string = format_metrics(hits_k, mrr, rmse)
        PRINT(f"evaluation [{run.current_epoch:>02d}]: \n" + metric_string)

        update_result = run.check_best_result(hits_k, map_k, mrr, rmse)

        if update_result:
            PRINT(f"update best score at epoch {run.best_epoch}")
            if args.save:
                log_results(args.save_file, metric_string, run.best_epoch)
            early_stopping_check = 0
        else:
            early_stopping_check += 1
            if early_stopping_check > config['patience']:
                PRINT(f"stop training at epoch {run.current_epoch}")
                break
        run.current_epoch += 1

    final_str = format_metrics(run.best_hits, run.best_mrr, run.valid_RMSE)
    PRINT(f"final results at epoch {run.best_epoch}:\n" + final_str)
