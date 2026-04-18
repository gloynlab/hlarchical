
import yaml
import torch
import os
from argparse import Namespace
from models.supports.utils import *
from models.train_model import ModelTrainer
import time

torch.backends.cudnn.deterministic = True

def run_train(
    args: Namespace,
    params,
    seed,
    logger,
    mode,
    data_dir,
    ref,
) -> str:
    """
    Args:
        args (Namespace):
        hps_name (str): Name of hyperparameter search target.
            must have keys listed in `MUST_KEYS`. 
        info_string (str):
    Returns:
        save_loc (str):
    """
    trainer = ModelTrainer(args, params, seed, logger, mode, data_dir, ref)
    acc = trainer.run()
    del trainer
    return acc

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--config', help='path to config file', 
        dest='config')

    parser.add_argument(
        '--data_dir', help='path to save results', 
        dest='data_dir')

    parser.add_argument(
        '--ref', help='reference prefix', 
        dest='ref')

    parser.add_argument('--device', default='cpu', dest='device')
    args = parser.parse_args()

    params = path_to_dict(args.config)
    os.makedirs(f'{args.data_dir}/{params["exp_name"]}')
    log_file = f'{args.data_dir}/{params["exp_name"]}/training.log'
    logger = Logger(log_file)

    if params['fold_num'] == -1:
        start_time = time.time()
        logger.log(f'process starts at {start_time}')
        _ = run_train(args, params, 0, logger, 'exp', args.data_dir, args.ref)
        end_time = time.time()
        logger.log(f'process ends at {end_time}')
        logger.log(f'all processing time is {end_time - start_time} seconds.')
    else:
        for seed in range(params['fold_num']):
            _ = run_train(args, params, seed, logger, 'exp', args.data_dir, args.ref)
