import argparse
from datetime import datetime
import os
import torch
from train import train_model
from configuration import build_config
from tensorboardX import SummaryWriter
import random 
import numpy as np


def train_classifier(run_id, use_cuda, args):
    cfg = build_config(args.dataset)
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(os.path.join(cfg.tf_logs_dir, str(run_id)))
    for arg in vars(args):
        writer.add_text(arg, str(getattr(args, arg)))
    train_model(cfg, run_id, save_dir, use_cuda, args, writer)


def main(args):
    print("Run description : ", args.run_description)

    # call a function depending on the 'mode' parameter
    if args.train_classifier:
        run_id = args.run_id + '_' + datetime.today().strftime('%d-%m-%y_%H%M')
        use_cuda = torch.cuda.is_available()

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        train_classifier(run_id, use_cuda, args)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train Multi-label Classification model')

    # 'mode' parameter (mutually exclusive group) with five modes : train/test classifier, train/test generator, test
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--train_classifier', dest='train_classifier', action='store_true',
                       help='Training the Classifier')

    parser.add_argument("--gpu", dest='gpu', type=str, required=False, help='Set CUDA_VISIBLE_DEVICES environment variable, optional')

    parser.add_argument('--run_id', dest='run_id', type=str, required=False, help='Please provide an ID for the current run')

    parser.add_argument('--run_description', dest='run_description', type=str, required=False, help='Please description of the run to write to log')

    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use.', choices=["ntu_rgbd_120", "small_ntu_rgbd_120", "pkummd", "pkummdv1", 'charades', 'mergedntupk', 'PCharades', 'tennis'])

    parser.add_argument('--model_version', type=str, required=True, help='Specify the model version to use for transformer.', 
                        choices=["baseline", "i3d", "r3d", "r2plus1d", "v1", "v2", 'v3', "vivit", 'v2+backbone', 'v3+backbone', 'v3+D', 'v3_intermediate', 'swin'])

    parser.add_argument('--input_type', type=str, required=True, help='Specify if the input is either RGB or Flow.', choices=["rgb", "ir", "skeleton"])

    parser.add_argument('--input_dim', type=int, default=224, help='Size of the frames in the input clip.')

    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames in the input clip.')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--skip', type=int, help='Number of frames to skip in the input sequence.')

    group.add_argument("--random_skip", action='store_true')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

    parser.add_argument('--steps', type=int, default=1, help='Number of accumulation steps.')

    parser.add_argument('--patch_size', type=int, default=16, help='Size of the patches for the input frames.')

    parser.add_argument('--hidden_dim', type=int, default=512, help='Size of the features in the Transformer Encoder Layer.')

    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the encoder layers.')

    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers for encoder/decoder in the transformer model.')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in the dataloader.')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the FC layers.')

    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay.')

    parser.add_argument('--optimizer', type=str, default='ADAM', help='provide optimizer preference')

    parser.add_argument('--f1_threshold', type=float, default=0.5, help='Probability threshold for computing F1 Score')

    parser.add_argument('--checkpoint', type=str, required=False, help='Path to the pre-trained model.')

    parser.add_argument('--ema-decay', default=0.999, type=float)

    parser.add_argument('--validation_interval', type=int, default=5, help='Number of epochs between validation step.')

    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    
    parser.add_argument('--action_train', type=bool, default=False, help='Train the baselines with action loss or not')
    
    parser.add_argument('--contrastive_train', type=bool, default=False, help='Train the baselines with contrastive feature loss or not')

    # parse arguments
    args = parser.parse_args()

    # set environment variables to use GPU-0 by default
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # exit when the mode is 'train_classifier' and the parameter 'run_id' is missing
    if args.train_classifier:
        if args.run_id is None:
            parser.print_help()
            exit(1)

    main(args)
