import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import EGGA_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/byh_data/SSNet_data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='EGGA', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')

args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

    "加载神经网络模型和优化器的状态"


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

    "更新模型的指数移动平均"


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

    "生成一个掩码，用于在训练过程中对图像进行部分遮挡。"


def generate_mask(img):
    batch_size, img_x, img_y = img.shape[0], img.shape[1], img.shape[2]  # 获取输入图像img的维度信息。
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    nonzero_indices = torch.nonzero(img)
    if nonzero_indices.size()[0] != 0:

        top_left = torch.min(nonzero_indices, dim=0)[0]
        bottom_right = torch.max(nonzero_indices, dim=0)[0]
        # print("非零区域的左上坐标:", top_left.tolist())
        # print("非零区域的右下坐标:", bottom_right.tolist())
        x1, y1 = top_left[1], top_left[2]
        x2, y2 = bottom_right[1], bottom_right[2]
        x1 = x1.data.cpu()
        x2 = x2.data.cpu()
        y1 = y1.data.cpu()
        y2 = y2.data.cpu()
        # print(x1,x2,y1,y2)
        patch_x, patch_y = int(img_x * 1 / 2), int(img_y * 1 / 2)
        # print(patch_x,patch_y)
        w = np.random.randint(x1, (x1 + x2) / 2)
        h = np.random.randint(y1, (y1 + y2) / 2)
        mask[w:w + patch_x, h:h + patch_y] = 0
        loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
        return mask.long(), loss_mask.long()
    else:
        patch_x, patch_y = int(img_x * 1 / 2), int(img_y * 1 / 2)
        w = np.random.randint(0, img_x - patch_x)
        h = np.random.randint(0, img_y - patch_y)
        mask[w:w + patch_x, h:h + patch_y] = 0
        loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
        return mask.long(), loss_mask.long()


def generate_mask1(lab):  # 该函数生成一个用于图像遮挡的掩码。
    batch_size, lab_x, lab_y = lab.shape[0], lab.shape[1], lab.shape[2]  # 获取输入图像img的维度信息。
    loss_mask = torch.ones(batch_size, lab_x, lab_y).cuda()  # 创建一个与输入图像相同大小的全1张量loss_mask，用于标记不需要进行遮挡的区域。
    mask = torch.ones(lab_x, lab_y).cuda()  # 创建一个与输入图像大小相同的全1张量mask，用于生成遮挡掩码。
    nonzero_indices = torch.nonzero(lab)
    if nonzero_indices.size()[0] != 0:
        top_left = torch.min(nonzero_indices, dim=0)[0]
        bottom_right = torch.max(nonzero_indices, dim=0)[0]

        print("非零区域的左上坐标:", top_left.tolist())
        print("非零区域的右下坐标:", bottom_right.tolist())
        x1, y1 = top_left[1], top_left[2]
        x2, y2 = bottom_right[1], bottom_right[2]
        w = x2 - x1
        h = y2 - y1
        # print(x1, y1, x2, y2)
        if w <= 128:
            if h <= 128:
                wd = int((128 - w) / 2)
                # print('wd:', wd)
                hd = int((128 - h) / 2)
                # print('hd:', hd)
                x_1, x_2, y_1, y_2 = x1 - wd, x1 - wd + 128, y1 - hd, y1 - hd + 128
                if x_1 < 0:
                    x_1, x_2 = 0, 128
                if y_1 < 0:
                    y_1, y_2 = 0, 128

                mask[x_1:x_2, y_1:y_2] = 0
                loss_mask[:, x_1:x_2, y_1:y_2] = 0
                # print(x_1, y_1, x_2, y_2)
            else:
                mask[x1:x1 + 128, y1:y1 + 128] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
                loss_mask[:, x1:x1 + 128, y1:y1 + 128] = 0
                x_1, x_2, y_1, y_2 = x1, x1 + 128, y1, y1 + 128
        else:
            mask[x1:x1 + 128, y1:y1 + 128] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
            loss_mask[:, x1:x1 + 128, y1:y1 + 128] = 0
            x_1, x_2, y_1, y_2 = x1, x1 + 128, y1, y1 + 128
            '''mask[x1:x1+w , y1:y1+h] = 0
            loss_mask[:, x1:x1+w , y1:y1+h] = 0
            x_1, x_2, y_1, y_2 = x1, x2, y1, y2'''
    else:
        patch_x, patch_y = int(lab_x * 1 / 2), int(lab_y * 1 / 2)  # 计算遮挡区域的大小为输入图像的2/3。
        w = np.random.randint(0, lab_x - patch_x)  # 随机生成遮挡区域的起始横坐标。
        h = np.random.randint(0, lab_y - patch_y)  # 随机生成遮挡区域的起始纵坐标。
        mask[w:w + patch_x, h:h + patch_y] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
        loss_mask[:, w:w + patch_x, h:h + patch_y] = 0  # 在loss_mask中标记对应的遮挡区域。
        x_1 = np.random.randint(0, lab_x - patch_x)
        y_1 = np.random.randint(0, lab_y - patch_y)
        x_2 = x_1 + 128
        y_2 = y_1 + 128

    return mask.long(), loss_mask.long(), x_1, y_1, x_2, y_2


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask

    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def change_mask(img_a, img_b, lab_a, lab_b):
    a_mask, a_loss_mask, xa1, ya1, xa2, ya2 = generate_mask1(lab_a)
    b_mask, b_loss_mask, xb1, yb1, xb2, yb2 = generate_mask1(lab_b)
    if xa2 <= 224 and ya2 <= 224 and xb2 <= 224 and yb2 <= 224:
        a_region = img_a[:, :, xa1:xa2, ya1:ya2].clone()

        b_region = img_b[:, :, xb1:xb2, yb1:yb2].clone()

        la_region = lab_a[:, xa1:xa2, ya1:ya2].clone()
        lb_region = lab_b[:, xb1:xb2, yb1:yb2].clone()

        img_a[:, :, xa1:xa2, ya1:ya2] = b_region

        img_b[:, :, xb1:xb2, yb1:yb2] = a_region
        lab_a[:, xa1:xa2, ya1:ya2] = lb_region
        lab_b[:, xb1:xb2, yb1:yb2] = la_region

    else:
        img_a = img_a
        img_b = img_b
        lab_a = lab_a
        lab_b = lab_b

    return img_a, img_b, lab_a, lab_b


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def diceloss(output, img_l):
    CE = nn.CrossEntropyLoss(reduction='mean')
    img_l = img_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1))
    loss_ce = CE(output, img_l)
    return loss_dice, loss_ce


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = EGGA_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            # if iter_num>=10000:
            img_mask, loss_mask = generate_mask(lab_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)

            out_mixl = model(net_input.cuda())
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            #             else:
            #                 img_maska, loss_maska, _, _, _, _ = generate_mask1(lab_a)  # 调用generate_mask函数生成图像遮挡的掩码img_mask和损失遮挡的掩码loss_mask
            #                 img_maskb, loss_maskb, _, _, _, _ = generate_mask1(lab_b)

            #                 cimg_a, cimg_b, clab_a, clab_b = change_mask(img_a, img_b, lab_a, lab_b)

            #                 net_input = torch.cat((cimg_a, img_a, cimg_b, img_b), dim=0)

            #                 gt_mixl = torch.cat((clab_a, clab_b), dim=0)

            #                 out_mixl = model(net_input.cuda())
            #                 # loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            #                 loss_dicea1c, loss_cea1c = diceloss(out_mixl[0:6],
            #                                                     clab_a)  # 调用mix_loss函数计算混合损失，其中使用模型输出out_mixl、有标记的标签lab_a、无标记的标签lab_b、损失遮挡掩码loss_mask，设置无标记数据权重为1.0，同时指定使用无标记数据。

            #                 loss_dicea2c, loss_cea2c = diceloss(out_mixl[6:12], clab_a)

            #                 loss_diceb1c, loss_ceb1c = diceloss(out_mixl[12:18], clab_b)
            #                 loss_diceb2c, loss_ceb2c = diceloss(out_mixl[18:24], lab_b)

            #                 loss_dice = (loss_dicea1c + loss_diceb1c) * 0.5 + loss_dicea2c + loss_diceb2c
            #                 loss_ce = (loss_cea1c + loss_ceb1c) * 0.5 + loss_cea2c + loss_ceb2c

            #                 loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))

    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = EGGA_net(in_chns=1, class_num=num_classes)
    "ema=true 指数平均移动"
    ema_model = EGGA_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[
                                                                                              args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                img_mask, loss_mask = generate_mask(lab_a)
                unl_label = ulab_a * img_mask + lab_a * (1 - img_mask)
                l_label = lab_b * img_mask + ulab_b * (1 - img_mask)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            # net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_unl = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l = img_b * img_mask + uimg_b * (1 - img_mask)
            out_unl = model(net_input_unl)
            out_l = model(net_input_l)
            unl_dice, unl_ce = mix_loss(out_unl, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_l, lab_b, plab_b, loss_mask, u_weight=args.u_weight)

            loss_ce = unl_ce + l_ce
            loss_dice = unl_dice + l_dice

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            # if iter_num % 20 == 0:
            #     image = net_input_unl[1, 0:1, :, :]
            #     writer.add_image('train/Un_Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(out_unl, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = unl_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/Un_GroundTruth', labs, iter_num)
            #
            #     image_l = net_input_l[1, 0:1, :, :]
            #     writer.add_image('train/L_Image', image_l, iter_num)
            #     outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
            #     labs_l = l_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/EGGA/ACDC3_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/EGGA/ACDC3_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('ACDC_EGGA_train.py', self_snapshot_path)
    shutil.copy('../code/networks/unet.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)




