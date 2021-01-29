import os
import sys
import time
from datetime import datetime
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

import tensorboardX

# Parse all arguments, important have note behind them
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SNAS_edge_all', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--meta_arch', type=str, default='6-6-6', help='meta-architecture')
parser.add_argument('--saved_model', type=str, default='none', help='location of saved model')
args = parser.parse_args()
print(args.arch)
args.save = 'eval-{}-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.meta_arch)
generate_date = str(datetime.now().date())
utils.create_exp_dir(generate_date,args.save, scripts_to_save=glob.glob('*.py'))

# Setup logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

logger = tensorboardX.SummaryWriter('./runs/eval_{}'.format(args.arch))


def main():
    # Check if GPU is available
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # Setup CUDA environment
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # Load in model with given genotype
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype, args.meta_arch)
    model = model.cuda()

    if args.saved_model != 'none':
        utils.load(model, args.saved_model)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    # Set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Load train en validation dataset
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    # Iterate over epochs
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # Train for one epoch
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch)
        logging.info('train_acc %f', train_acc)
        logger.add_scalar("epoch_train_acc", train_acc, epoch)

        # Get validation accuracy after epoch
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        logger.add_scalar("epoch_valid_acc", valid_acc, epoch)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

# Train the model for one epoch
def train(train_queue, model, criterion, optimizer, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    # Iterate over all batches in training queue
    for step, (input, target) in enumerate(train_queue):
        # Get input and target from train queue
        input = Variable(input).cuda()
        target = Variable(target).cuda(non_blocking=True)

        # Forward
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)

        # Check if auxiliary head is added, if so, add auxiliary loss to total loss
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        # Optimize
        optimizer.step()

        # Calculate top1 and top5 accuracy
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # Log training info if necessary
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logger.add_scalar("iter_train_loss", objs.avg, step + len(train_queue.dataset) * epoch)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        # Iterate over validation queue
        for step, (input, target) in enumerate(valid_queue):
            # Load input and target from validation queue
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            # Calculate top1 and top5 accuracy of validation set
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # Log validation info if necessary
            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
