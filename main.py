import argparse
import random

import numpy as np
import torch

from utils.tools import loading_pems_graph


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed=None):
    if seed is None:
        seed = torch.randint(999999, (1,)).item()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate MobiMixer for mobile traffic prediction."
    )

    # Data
    parser.add_argument("--data", type=str, default="mobility_traffic", help="dataset type")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/mobility_traffic/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="mobility_traffic.csv", help="data file")
    parser.add_argument("--graph_path", type=str, default="mobility_traffic.npy", help="graph file")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help=(
            "time feature frequency, e.g. s, t, h, d, b, w, m, "
            "or detailed values such as 15min and 3h"
        ),
    )
    parser.add_argument("--features", type=str, default="M", help="forecasting task type")

    # Experiment metadata
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model_id", type=str, default="MobiMixer")
    parser.add_argument("--model", type=str, default="MobiMixer")
    parser.add_argument("--des", type=str, default="test", help="experiment description")
    parser.add_argument("--comment", type=str, default="none", help="extra experiment comment")
    parser.add_argument("--itr", type=int, default=1, help="experiment repeats")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    # Sequence lengths
    parser.add_argument("--node_num", type=int, default=10000, help="number of input nodes")
    parser.add_argument("--seq_len", type=int, default=24, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=24, help="start token length")
    parser.add_argument("--pred_len", type=int, default=244, help="prediction sequence length")

    # Model
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=16, help="model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=32, help="feed-forward dimension")
    parser.add_argument("--moving_avg", type=int, default=25, help="moving average window")
    parser.add_argument("--factor", type=int, default=1, help="attention factor")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time feature encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--top_k", type=int, default=5, help="top-k period selection")
    parser.add_argument("--num_kernels", type=int, default=6, help="number of kernels")
    parser.add_argument("--distil", action="store_true", help="use distillation")
    parser.add_argument("--output_attention", action="store_true", help="return attention")
    parser.add_argument("--channel_independent", action="store_true")
    parser.add_argument("--individual", action="store_true")

    # Multi-scale processing
    parser.add_argument("--down_sampling_layers", type=int, default=1)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg", choices=["avg", "max", "none"])
    parser.add_argument("--only_use_down_sampling", action="store_true")
    parser.add_argument("--pred_down_sampling", action="store_true")

    # Graph options
    parser.add_argument("--use_graph", type=int, default=0)
    parser.add_argument("--graph_mask", action="store_true")

    # Optimization
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="checkpoint location")
    parser.add_argument("--train_epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="learning-rate adjustment")
    parser.add_argument("--pct_start", type=float, default=0.2, help="OneCycleLR pct_start")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision")

    # Hardware
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="use GPU when available")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple GPUs")
    parser.add_argument("--devices", type=str, default="1", help="comma-separated multi-GPU ids")
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector",
    )
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="projector hidden layers")

    return parser


def build_setting(args, iteration):
    return (
        f"{args.data_path}_{args.task_name}_{args.model_id}_{args.comment}_"
        f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_"
        f"df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_"
        f"{args.des}_{iteration}"
    )


def main():
    args = build_parser().parse_args()
    args.seed = set_seed(args.seed)
    args.use_gpu = torch.cuda.is_available() and args.use_gpu

    from mobility_traffic import Mobility_Traffic

    if args.down_sampling_method == "none":
        args.down_sampling_method = None

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    for iteration in range(args.itr):
        setting = build_setting(args, iteration)
        exp = Mobility_Traffic(args)

        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        loading_pems_graph(args)
        exp.train(setting)

        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        loading_pems_graph(args)
        exp.test(setting)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
