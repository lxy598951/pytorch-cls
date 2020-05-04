# -*- coding: utf-8 -*-
from __future__ import print_function, division

from archs.MobileNetV1 import MobileNetV1
from archs.MobileNetV1_Pool import MobileNetV1_Pool
from archs.MobileNetV3 import MobileNetV3_Large, MobileNetV3_Small
from archs.MobileNetV3_Pool import MobileNetV3_Large_Pool, MobileNetV3_Small_Pool

from archs.mnv2 import MobileNetV2
from archs.mnv2_1x1 import MobileNetV2_1x1
from archs.mnv2_3x3 import MobileNetV2_3x3
from archs.mnv2_3x3_product import MobileNetV2_3x3_Product
from archs.se_mnv2_3x3 import SE_MobileNetV2_3x3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import pdb
import argparse
import logging
from logging import handlers
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(
            self,
            filename,
            level='info',
            when='D',
            backCount=3,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,
                                               when=when,
                                               backupCount=backCount,
                                               encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def eval_model(model, val_dataloader, criterion, logs=None):
    logs.logger.info('Begin evaluating process...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inference_time = list()

    running_loss = 0.0
    running_corrects = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward
            start = time.time()
            outputs = model.forward(inputs)
            inference_time.append(time.time() - start)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
        average_inference_time = sum(inference_time) / len(inference_time)

    return epoch_loss, epoch_acc, average_inference_time


def train_model(model,
                data_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25,
                logs=None,
                args=None):
    logs.logger.info('Begin training process...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(args.ckpt, args.data_type)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_loader, val_loader = data_loader['train'], data_loader['val']

    best_model_wts = model.state_dict()
    best_acc = 0.0

    inference_time = list()
    for epoch in range(num_epochs):
        logs.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logs.logger.info('-' * 10)

        model.train()
        # scheduler.step()

        running_loss = 0.0
        running_corrects = 0.0

        print_every = 0

        for inputs, labels in train_loader:
            # To cuda
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            start = time.time()
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            print_every += args.per_batch_size * args.ngpus
            current_loss = loss.item() * inputs.size(0)
            current_corrects = torch.sum(preds == labels.data)
            running_loss += current_loss
            running_corrects += current_corrects
            if print_every // (args.per_batch_size * args.ngpus) == 20:
                print_every = 0
                logs.logger.info('Current running loss is {:.4f}'.format(
                    current_loss / (args.per_batch_size * args.ngpus)))
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        logs.logger.info('{} Loss: {:.4f} Acc: {:.2f}%'.format(
            'train', train_loss, train_acc * 100))

        logs.logger.info('Starting evaluation...')
        val_loss, val_acc, avg_infer_time = eval_model(model, val_loader,
                                                       criterion, logs)
        logs.logger.info('{} Loss: {:.4f} Acc: {:.2f}%'.format(
            'val', val_loss, val_acc * 100))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir,
                            args.arch + '-{:.4f}'.format(val_acc.data) + '.pth'))
        inference_time.append(avg_infer_time)

    average_inference_time = sum(inference_time) / len(inference_time)
    logs.logger.info(
        'Average inference time: {:7f}'.format(average_inference_time))
    logs.logger.info('Best val Acc: {:4f}%'.format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='mb1')
    parser.add_argument('--data_type', type=str, default='imagenet')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--per_batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--ckpt', type=str, default='./ckpt')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()

    log_dir = os.path.join(args.log_path, args.data_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(
            os.path.join(log_dir, args.arch + '_' + args.data_type + '.log')):
        os.remove(
            os.path.join(log_dir, args.arch + '_' + args.data_type + '.log'))
    logs = Logger(os.path.join(log_dir,
                               args.arch + '_' + args.data_type + '.log'),
                  level='debug')

    logs.logger.info('Loading Dataset...')
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':
        transforms.Compose([
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(args.data_path, x),
                                data_transforms[x])
        for x in ['train', 'val']
    }
    # wrap your data and label into Tensor
    dataloders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=args.per_batch_size *
                                       args.ngpus,
                                       shuffle=True,
                                       num_workers=16,
                                       pin_memory=True)
        for x in ['train', 'val']
    }

    logs.logger.info('Loading model...')
    # pdb.set_trace()
    if args.arch == 'mb1':
        model_ft = MobileNetV1(args.classes)
    elif args.arch == 'mb1p':
        model_ft = MobileNetV1_Pool(args.classes)
    elif args.arch == 'mb2':
        model_ft = MobileNetV2(args.classes)
    elif args.arch == 'mb2p_1x1':
        model_ft = MobileNetV2_1x1(args.classes)
    elif args.arch == 'mb2p_3x3':
        model_ft = MobileNetV2_3x3(args.classes)
    elif args.arch == 'mb2p_3x3_product':
        model_ft = MobileNetV2_3x3_Product(args.classes)
    elif args.arch == 'se_mb2p_3x3':
        model_ft = SE_MobileNetV2_3x3(args.classes)
    # elif args.arch == 'mb2p_dilation':
    #     model_ft = MobileNetV2_3x3_dilation(args.classes)
    # elif args.arch == 'se_mb2_at_dilation':
    #     model_ft = SE_MobileNetV2_3x3_dilation(args.classes)
    elif args.arch == 'mb3s':
        model_ft = MobileNetV3_Small(args.classes)
    elif args.arch == 'mb3l':
        model_ft = MobileNetV3_Large(args.classes)
    elif args.arch == 'mb3sp':
        model_ft = MobileNetV3_Small_Pool(args.classes)
    else:
        model_ft = MobileNetV3_Large_Pool(args.classes)

    # Use gpu or not
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_gpu else torch.device('cpu')
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in range(args.ngpus)])
        if args.ngpus > 1:
            device_ids = [i for i in range(args.ngpus)]
            model_ft = nn.DataParallel(model_ft, device_ids=device_ids)
    model_ft.to(device)

    # Define loss_fn
    criterion = nn.CrossEntropyLoss().to(device)

    # Observe that all parameters are being optimized
    if args.optimizer == 'rmsprop':
        optimizer_ft = optim.RMSprop(model_ft.parameters(), 
            lr=args.lr,
            momentum=0.9,
            weight_decay=4e-5)
    elif args.optimizer == 'sgd':
        optimizer_ft = optim.SGD(model_ft.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=4e-5)

    # cosine lr scheduler
    lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=args.epochs
    )
    # pdb.set_trace()
    model_ft = train_model(model=model_ft,
                           data_loader=dataloders,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=lr_scheduler,
                           num_epochs=args.epochs,
                           logs=logs,
                           args=args)
