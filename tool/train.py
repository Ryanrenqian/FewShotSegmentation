import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model import *
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


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


# def worker_init_fn(worker_id):
#     random.seed(args.manual_seed + worker_id)


def main_worker(argss):
    global args
    args = argss
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

    optimizer = model._optimizer(args)
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    # model
    model = torch.nn.DataParallel(model).cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, shot=args.shot, max_sp=args.max_sp, data_root=args.data_root, \
                                 data_list=args.train_list, transform=train_transform, mode='train', \
                                 use_coco=args.use_coco, use_split_coco=args.use_split_coco)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    if args.evaluate:
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
        val_data = dataset.SemData(split=args.split, shot=args.shot, max_sp=args.max_sp, data_root=args.data_root, \
                                   data_list=args.val_list, transform=val_transform, mode='val', \
                                   use_coco=args.use_coco, use_split_coco=args.use_split_coco)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    max_iou = 0.
    filename = 'ASGNet.pth'
    folder = os.path.join(os.path.dirname(args.save_path), 'bestpractice')
    npzfile = os.path.join(folder, 'protos.npz')
    os.system(f'mkdir -p {folder}')
    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        loss_train, aux_loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        # if main_process():
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('aux_loss_train', aux_loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if args.evaluate and (epoch % 2 == 0 or (epoch <= 70 and epoch % 1 == 0)):
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, class_iou_class, protos, labels, aux_protos = validate(
                val_loader, model, criterion)
            # if main_process():
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('class_miou_val', class_miou, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou:
                np.savez(npzfile, labels=labels, protos=protos, aux_protos=aux_protos)
                max_iou = class_miou
                max_class_iou_class = class_iou_class
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           filename)

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
    logger.info('Best mIOU: : ' + str(max_iou))
    for i in range(len(max_class_iou_class)):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, max_class_iou_class[i]))


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    best_iou = 0
    vis_key = 0
    print('Warmup: {}'.format(args.warmup))
    for i, (input, target, s_input, s_mask, s_init_seed, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                               index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        s_mask = torch.where(s_mask == 1, s_mask, torch.zeros_like(s_mask))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = torch.where(target == 1, target, torch.zeros_like(target))
        s_init_seed = s_init_seed.cuda(non_blocking=True)

        output, main_loss, aux_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target, s_seed=s_init_seed)

        main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        n = input.size(0)
        if np.isnan(loss.item()):
            raise ValueError('Loss value is NaN!')
        if np.isnan(main_loss.item()):
            raise ValueError('main_loss value is NaN!')
        if np.isnan(aux_loss.item()):
            raise ValueError('aux_loss value is NaN!')
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)

        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        # if main_process():
        writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
        writer.add_scalar('aux_loss_train_batch', aux_loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # if main_process():
    logger.info(
        'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc,
                                                                                       allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    return main_loss_meter.avg, aux_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
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
            loss = criterion(output, target)
            
            total_time = total_time + 1
            model_time.update(time.time() - start_time)
            #
            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]
            protos.append(proto.detach().cpu().numpy()[0])
            if aux_proto is not None:
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
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, class_iou_class, protos, labels, aux_protos


if __name__ == '__main__':
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpu
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
