import logging
import argparse
import os
import random
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def parse_arguments(debug=False):
    def str2list(input_str):
        if not input_str:
            return []
        return input_str.replace(" ", "").split(",")

    def str2bool(input_str):
        input_str = input_str.strip().lower()
        if input_str in ("true", "y", "yes", "1"):
            return True
        elif input_str in ("false", "n", "no", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    dataset_dict = {
        "beauty": "Beauty",
        "clothing": "Clothing_Shoes_and_Jewelry",
        "sports": "Sports_and_Outdoors",
        "toys": "Toys_and_Games",
        "home": "Home_and_Kitchen",
        "phones": "Cell_Phones_and_Accessories",
        "ml-1m": "ml-1m",
    }

    parser = argparse.ArgumentParser()
    #################### GLOBAL ####################
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seed", type=int, default=1234, help="Seed value for reproducibility.")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    parser.add_argument("--exp_name", type=str, default=datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--data_dir", type=str, default="data/processed", help="Directory where data files are located."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saved_model_path", type=str, default=None, help="Path to a pre-trained model for testing.")

    #################### DATA ####################
    parser.add_argument("--dataset", type=str, default="beauty", choices=dataset_dict.keys())
    parser.add_argument("--feature_cols", type=str2list, default="image,text", help="Feature columns to be used.")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="Batch size for training data.")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size for evaluation data.")
    parser.add_argument("--maxlen", type=int, default=50, help="Maximum sequence length.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    #################### MODEL ####################
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dim_head", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--feature_transformer", type=str, default="conditional", choices=["full", "conditional", "none"]
    )
    parser.add_argument("--feature_dropout", type=float, default=0.5)
    parser.add_argument("--feature_num_experts", type=int, default=4)
    parser.add_argument("--feature_num_routed_queries", type=str2list, default="16,16,16,16")
    parser.add_argument("--feature_num_routed_key_values", type=str2list, default="16,16,16,16")
    parser.add_argument("--item_transformer", type=str, default="conditional", choices=["full", "conditional", "none"])
    parser.add_argument("--item_num_experts", type=int, default=4)
    parser.add_argument("--item_num_routed_queries", type=str2list, default="8,8,8,8")
    parser.add_argument("--item_num_routed_key_values", type=str2list, default="8,8,8,8")
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument("--ff_dropout", type=float, default=0.2)
    parser.add_argument("--use_flash", type=str2bool, default="no")
    parser.add_argument("--null_token_to_unrouted", type=str2bool, default="no")

    #################### TRAIN & EVALUATE ####################
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--lr_warmup_step", type=int, default=4, help="Number of warmup steps for the learning rate.")
    parser.add_argument(
        "--lr_milestones", type=str2list, default="", help="Milestones for the learning rate scheduler."
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=1.0,
        help="Factor by which the learning rate will be reduced when hitting a milestone.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay factor for regularization.")
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument("--eval_step", type=int, default=1, help="Frequency of evaluation in terms of epochs.")
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--label_gamma", type=float, default=0.5)

    args = parser.parse_args([]) if debug else parser.parse_args()
    args.dataset = dataset_dict[args.dataset.lower()]

    elem_to_int = lambda l: [int(n) for n in l] if type(l) == list else int(l)
    args.feature_num_routed_queries = elem_to_int(args.feature_num_routed_queries)
    args.feature_num_routed_key_values = elem_to_int(args.feature_num_routed_key_values)
    args.item_num_routed_queries = elem_to_int(args.item_num_routed_queries)
    args.item_num_routed_key_values = elem_to_int(args.item_num_routed_key_values)
    args.lr_milestones = elem_to_int(args.lr_milestones)

    return args


def get_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)d] %(message)s")
    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = f"{datetime.strftime(datetime.now(), '%Y%m%d_%H%M')}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


def get_writer(args, exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        f.writelines([f"{k}: {v}\n" for k, v in vars(args).items()])
    writer = SummaryWriter(exp_dir)
    return writer


def set_seed(seed, logger):
    logger.info(f"Set Seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
