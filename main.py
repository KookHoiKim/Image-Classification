import argparse
import os

# utils
import errno
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith('__')
        and callable(models.__dict__[name])
        )

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout',
                    help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str,
                    help='path to latest checkpoint (default : none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to last checkpoint (default : none)')

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default : resnet18)')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
#if args.manualSeed is None:
#    args.manualSeed = 950103
#random.seed(args.manualSeed)
#torch.manual_seed(args.manualSeed)
#if use_cuda:
#    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0    # best test accuracy

# temporal function, move to utils later
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errono.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def main():
    global best_acc
    start_epoch = args.start_epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading
#    traindir = os.path.join(args.data, 'train')
#    valdir = os.path.join(args.data, 'val')
    #normalize = transforms.Normalize()

    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        ),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    ''' 
    trainset = datasets.ImageNet('./imagenet_data', train=True, download=True)
    trainloader = data.DataLoader(trainset, shuffle=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()









if __name__ == '__main__':
    main()

