import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import pickle
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
# from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
from model import *
from util import dataset
from util import transform, config
from util.util import AverageMeter, intersectionAndUnionGPU
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from glob import glob


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
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


def save_fig(origin, logits, target, save_path):
    '''
    target: Tensor
    logits: Tensor
    '''
    #     transform = ToPILImage()
    b, c, w, h = origin.shape

    heatmap = np.uint8(np.clip(logits.detach().cpu().softmax(1)[:, 1, :, :], 0, 1) * 255).transpose((1, 2, 0))
    mask = np.uint8(np.clip(logits.detach().cpu().max(1)[1], 0, 1) * 255).transpose((1, 2, 0))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    target = np.uint8(target.detach().cpu().numpy() * 255).transpose((1, 2, 0))
    origin_image = origin.cpu().squeeze(0).numpy().transpose((1, 2, 0))
    #     origin_image = np.transpose(origin_image, (1, 2, 0))
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    if origin_image.min() <= 0:
        origin_image *= std
        origin_image += mean

    if origin_image.min() <= 0:
        origin_image *= std
        origin_image += mean
    #     print(heatmap.shape,origin_image.size)
    origin_image = np.uint8(origin_image * 255)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    heat_img = heatmap * 0.3 + np.asarray(origin_image) * 0.8
    mask_img = mask * 0.3 + np.asarray(origin_image) * 0.8
    gt_img = target * 0.3 + np.asarray(origin_image) * 0.8
    #     mask_img = cv2.cvtColor(np.uint8(mask_img), cv2.COLOR_RGB2BGR)
    #     heat_img = cv2.cvtColor(heat_img, cv2.COLOR_RGB2BGR)
    #     gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
    #     origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path + 'maskimg.jpg', mask_img)
    cv2.imwrite(save_path + 'heatimg.jpg', heat_img)
    cv2.imwrite(save_path + 'gt.jpg', gt_img)
    cv2.imwrite(save_path + 'orgin.jpg', origin_image)


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

    global logger
    logger = get_logger()
    # writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])

    # imgs_path = os.path.join(args.save_path), '../prediction/')

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            filelist = glob.glob(os.path.join(args.save_path, '*.pth'))
            if len(filelist) < 2:
                logger.info("=> no weight found at '{}'".format(args.weight))
                raise Exception("'no weight found at '{}'".format(args.weight))
            for i in filelist:
                if 'final' not in i:
                    checkpoint = torch.load(i)
                    logger.info("=> loaded weight '{}'".format(i))
                    break

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
    loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, class_iou_class, protos, labels, aux_protos = validate(
        val_loader, model, criterion, args)


def validate(val_loader, model, criterion, args):
    folder = os.path.join(os.path.dirname(args.save_path), 'bestpractice')
    npzfile = os.path.join(folder, 'protos.npz')
    os.system(f'mkdir -p {folder}')

    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    protos = []
    labels = []
    aux_protos = []
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

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
            test_num = 2000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    image_transform = ToPILImage()
    count = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, s_init_seed, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)
            # segment foreground
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()
            output, proto, aux_proto = model(s_x=s_input, s_y=s_mask, x=input, y=target, s_seed=s_init_seed)

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            subcls = subcls[0].cpu().numpy()[0]
            loss = criterion(output, target)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)
            #
            n = input.size(0)
            loss = torch.mean(loss)
            save_fig(input, output, target, os.path.join(folder, f'class{subcls}{count}'))
            count += 1
            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            protos.append(proto.detach().cpu().numpy()[0])
            if not aux_proto is None:
                aux_protos.append(aux_proto.detach().cpu().numpy()[0])
            labels.append(subcls)
            class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
            class_union_meter[(subcls - 1) % split_gap] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num / 100) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

        # if main_process():
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    np.savez(npzfile, labels=labels, protos=protos, aux_protos=aux_protos)
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, class_iou_class, protos, labels, aux_protos


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
