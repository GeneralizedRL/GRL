import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--preprocess', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--data', default='UMLS', type=str)
parser.add_argument('--embmodel', default='conve', type=str)
parser.add_argument('--few_prob', type=float, default='0.5')
parser.add_argument('--lr', type=float, default='0.003')
parser.add_argument('--lossweight', type=float, default='0.01')
parser.add_argument('--represent_mode', type=int, default=1,
                    help='-:1,concat:2,*:3')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--gpu', type=str, default="0")

args = parser.parse_args()