import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from model import *
from util import dataset
from util import transform, config
from util.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--best',type=str,help='best pfh path')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_worker(argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = eval(args.arch).Model(args)
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])

    imgs_path = os.path.join(args.save_path), '../prediction/')

    if args.weight:
        if os.path.isfile(args.best):
            logger.info("=> loading weight '{}'".format(args.best))
            checkpoint = torch.load(args.best)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.best))
        else:
            logger.info("=> no weight found at '{}'".format(args.best))
            raise Exception("'no weight found at '{}'".format(args.best))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split, shot=args.shot, max_sp=args.max_sp, data_root=args.data_root,
                               data_list=args.val_list, transform=val_transform, mode='val',
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=None)
    loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion, args)

def validate(val_loader, model, criterion, args):
    # if main_process():
    #     logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 1000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    for e in range(20):
        for i, (input, target, s_input, s_mask, s_init_seed, subcls, ori_label) in enumerate(val_loader):
            if (iter_num-1) * args.batch_size_val >= test_num:
                break

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(s_x=s_input, s_y=s_mask, x=input, y=target, s_seed=s_init_seed)


if __name__ == '__main__':
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    # torch.cuda.set_device(device=args.train_gpu)
    main_worker(args)
