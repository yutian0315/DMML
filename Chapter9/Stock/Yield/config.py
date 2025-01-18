import argparse
import torch
import numpy as np
import random

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of Stock')
        parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=0.0001, type=float, help='initial weight decay')
        parser.add_argument('--epoch', default=30, type=int, help='number of epochs for training')
        parser.add_argument('--input_size', default=8, type=int, help='number of input_size')
        parser.add_argument('--hidden_size', default=32, type=int, help='number of hidden_size')
        parser.add_argument('--output_size', default=1, type=int, help='number of output_size')
        parser.add_argument('--num_workers', default=4, type=int, help='number of num_workers')
        parser.add_argument('--sequence_length', default=5, type=int, help='number of sequence_length')
        parser.add_argument('--batch_size', default=64, type=int, help='number of batch_size')
        parser.add_argument('--seed', type=int, default=2024, help='random seed to set')
        args = parser.parse_args()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args

    def initialize(self):
        self.set_seed(self.args.seed)
        return self.args

    def set_seed(self, seed=1000):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False