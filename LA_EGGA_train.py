from asyncore import write
import importlib
import os
from sre_parse import SPECIAL_CHARS
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.EGGA_util import context_mask, mix_loss, parameter_sharing, update_ema_variables

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/byh_data/SSNet_data/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='EGGA', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=4, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of EGGA
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')


def compute_sobel_edges(input):
    """
    Compute Sobel edges for a batch of images.
    """
    edges_x = sobel(input, axis=-1)
    edges_y = sobel(input, axis=-2)
    edges_z = sobel(input, axis=-3)
    edges = np.sqrt(edges_x ** 2 + edges_y ** 2 + edges_z ** 2)
    return edges


def compute_distance_transform(binary_images):
    """
    Compute distance transform for a batch of binary images.
    """
    distances = []
    for img in binary_images:
        distance_map = distance_transform_edt(1 - img)
        distances.append(distance_map)
    return np.stack(distances)


def boundary_loss(pred, target, weight=0.1):
    """
    Compute boundary loss between predictions and targets with a given weight.

    Args:
        pred (torch.Tensor): Predicted tensor with shape (N, C, D, H, W).
        target (torch.Tensor): Target tensor with shape (N, C, D, H, W).
        weight (float): Weight for the boundary loss.

    Returns:
        torch.Tensor: The weighted boundary loss.
    """
    pred = pred.sigmoid()

    # Convert tensors to numpy arrays for edge computation
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Compute edges using Sobel filter
    pred_boundary_np = compute_sobel_edges(pred_np)
    target_boundary_np = compute_sobel_edges(target_np)

    # Convert back to tensors
    pred_boundary = torch.from_numpy(pred_boundary_np).to(pred.device)
    target_boundary = torch.from_numpy(target_boundary_np).to(pred.device)

    # Clamp boundaries to [0, 1]
    pred_boundary = torch.clamp(pred_boundary, 0, 1)
    target_boundary = torch.clamp(target_boundary, 0, 1)

    # Compute distance transform on target boundary
    target_boundary_dist = compute_distance_transform(target_boundary_np)
    target_boundary_dist = torch.from_numpy(target_boundary_dist).to(pred.device)

    # Calculate the boundary loss
    boundary_loss = torch.abs(pred_boundary - target_boundary) * target_boundary_dist

    # Return the mean loss multiplied by the given weight
    return weight * boundary_loss.mean()


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def generate_mask1(lab):
    batch_size, lab_x, lab_y, lab_z = lab.shape[0], lab.shape[1], lab.shape[2], lab.shape[3]
    loss_mask = torch.ones(batch_size, lab_x, lab_y, lab_z).cuda()
    mask = torch.ones(lab_x, lab_y, lab_z).cuda()
    nonzero_indices = torch.nonzero(lab)

    if nonzero_indices.size()[0] != 0:
        top_left = torch.min(nonzero_indices, dim=0)[0]
        bottom_right = torch.max(nonzero_indices, dim=0)[0]

        x1, y1, z1 = top_left[1], top_left[2], top_left[3]
        x2, y2, z2 = bottom_right[1], bottom_right[2], bottom_right[3]

        w = x2 - x1
        h = y2 - y1
        d = z2 - z1

        if w <= 112 and h <= 112 and d <= 80:
            wd = max(0, int((112 - w) / 2))
            hd = max(0, int((112 - h) / 2))
            dd = max(0, int((80 - d) / 2))

            x_1, x_2 = x1 - wd, x1 - wd + 112
            y_1, y_2 = y1 - hd, y1 - hd + 112
            z_1, z_2 = z1 - dd, z1 - dd + 80

            if x_1 < 0:
                x_1, x_2 = 0, 112
            if y_1 < 0:
                y_1, y_2 = 0, 112
            if z_1 < 0:
                z_1, z_2 = 0, 80

            mask[x_1:x_2, y_1:y_2, z_1:z_2] = 0
            loss_mask[:, x_1:x_2, y_1:y_2, z_1:z_2] = 0

        else:
            x_2, y_2, z_2 = x1 + 112, y1 + 112, z1 + 80
            mask[x1:x_2, y1:y_2, z1:z_2] = 0
            loss_mask[:, x1:x_2, y1:y_2, z1:z_2] = 0

        return mask.long(), loss_mask.long(), x1, y1, z1, x_2, y_2, z_2

    else:
        patch_x, patch_y, patch_z = int(lab_x * 1 / 2), int(lab_y * 1 / 2), int(lab_z * 1 / 2)
        w = np.random.randint(0, lab_x - patch_x)
        h = np.random.randint(0, lab_y - patch_y)
        d = np.random.randint(0, lab_z - patch_z)

        mask[w:w + patch_x, h:h + patch_y, d:d + patch_z] = 0
        loss_mask[:, w:w + patch_x, h:h + patch_y, d:d + patch_z] = 0

        x1 = np.random.randint(0, lab_x - 112)
        y1 = np.random.randint(0, lab_y - 112)
        z1 = np.random.randint(0, lab_z - 80)

        x_2 = x1 + 112
        y_2 = y1 + 112
        z_2 = z1 + 80

        return mask.long(), loss_mask.long(), x1, y1, z1, x_2, y_2, z_2


def change_mask(img_a, img_b, lab_a, lab_b):
    a_mask, a_loss_mask, xa1, ya1, za1, xa2, ya2, za2 = generate_mask1(lab_a)
    b_mask, b_loss_mask, xb1, yb1, zb1, xb2, yb2, zb2 = generate_mask1(lab_b)

    if (xa2 <= img_a.shape[2] and ya2 <= img_a.shape[3] and za2 <= img_a.shape[4] and
            xb2 <= img_b.shape[2] and yb2 <= img_b.shape[3] and zb2 <= img_b.shape[4]):

        # Clone regions
        a_region = img_a[:, :, xa1:xa2, ya1:ya2, za1:za2].clone()
        b_region = img_b[:, :, xb1:xb2, yb1:yb2, zb1:zb2].clone()

        # Clone labels
        la_region = lab_a[:, xa1:xa2, ya1:ya2, za1:za2].clone()
        lb_region = lab_b[:, xb1:xb2, yb1:yb2, zb1:zb2].clone()

        # Adjust tensor shapes to match target sizes
        a_region = F.interpolate(a_region, size=(112, 112, 76), mode='trilinear', align_corners=False)
        b_region = F.interpolate(b_region, size=(112, 112, 76), mode='trilinear', align_corners=False)

        # Check and assign regions back to images
        if (xa2 - xa1 == xb2 - xb1 == 112 and ya2 - ya1 == yb2 - yb1 == 112 and za2 - za1 == zb2 - zb1 == 76):
            img_a[:, :, xa1:xa2, ya1:ya2, za1:za2] = b_region
            img_b[:, :, xb1:xb2, yb1:yb2, zb1:zb2] = a_region
            lab_a[:, xa1:xa2, ya1:ya2, za1:za2] = lb_region
            lab_b[:, xb1:xb2, yb1:yb2, zb1:zb2] = la_region
        else:
            print("Sizes of regions do not match expected dimensions after interpolation.")

    return img_a, img_b, lab_a, lab_b


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                  :args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(lab_a, args.mask_ratio)
                # img_mask, loss_mask, _, _, _, _, _, _ = generate_mask1(
                #     lab_a)  # 调用generate_mask函数生成图像遮挡的掩码img_mask和损失遮挡的掩码loss_mask
            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            # img_maskb, loss_maskb, _, _, _, _, _,_= generate_mask1(lab_b)
            #
            # cimg_a, cimg_b, clab_a, clab_b = change_mask(img_a, img_b, lab_a, lab_b)

            outputs, _ = model(volume_batch)
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = (loss_ce + loss_dice) / 2

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num >= 300 and iter_num % 100 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
        param.detach_()  # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)

    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[
                                                                                       args.labeled_bs + sub_bs:]
            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                img_mask, loss_mask = context_mask(lab_a, args.mask_ratio)
                # img_mask, loss_mask, _, _, _, _, _, _ = generate_mask1(
                #     lab_a)  # 调用generate_mask函数生成图像遮挡的掩码img_mask和损失遮挡的掩码loss_mask
                # img_maskb, loss_maskb, _, _, _, _, _, _ = generate_mask1(lab_b)

                # img_a, img_b, lab_a, lab_b = change_mask(img_a, img_b, lab_a, lab_b)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            outputs_l, _ = model(mixl_img)
            outputs_u, _ = model(mixu_img)
            loss_l = mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            loss_u = mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)

            loss = loss_l + loss_u

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f' % (iter_num, loss, loss_l, loss_u))

            update_ema_variables(model, ema_model, 0.99)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 100 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = outputs_l.size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)

                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                outputs_l_sharpened = sharpening(outputs_l_soft)
                seg_out = outputs_l_sharpened[0, 1, ...].permute(2, 0, 1)  # y
                target = mixl_lab[0, ...].permute(2, 0, 1)
                train_img = mixl_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = mixu_lab[0, ...].permute(2, 0, 1)
                train_img = mixu_img[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                        torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/EGGA/LA8_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/EGGA/LA8_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting EGGA training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    # shutil.copy('../code/LA_EGGA_train.py', self_snapshot_path)
    # shutil.copy('../code/networks/VNet.py', self_snapshot_path)

    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)


