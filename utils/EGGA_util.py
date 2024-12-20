from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel
from skimage import segmentation as skimage_seg
import time

DICE = mask_DiceLoss(nclass=2)
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


import torch
import numpy as np


def generate_mask1(lab):  # 该函数生成一个用于图像遮挡的掩码。
    batch_size, lab_z, lab_x, lab_y = lab.shape  # 获取输入图像lab的维度信息。
    loss_mask = torch.ones(batch_size, lab_z, lab_x, lab_y).cuda()  # 创建一个与输入图像相同大小的全1张量loss_mask，用于标记不需要进行遮挡的区域。
    mask = torch.ones(lab_z, lab_x, lab_y).cuda()  # 创建一个与输入图像大小相同的全1张量mask，用于生成遮挡掩码。
    nonzero_indices = torch.nonzero(lab)
    if nonzero_indices.size(0) != 0:
        top_left = torch.min(nonzero_indices, dim=0)[0]
        bottom_right = torch.max(nonzero_indices, dim=0)[0]
        # 打印结果
        # print("非零区域的左上坐标:", top_left.tolist())
        # print("非零区域的右下坐标:", bottom_right.tolist())
        z1, x1, y1 = top_left[1], top_left[2], top_left[3]
        z2, x2, y2 = bottom_right[1], bottom_right[2], bottom_right[3]
        dz = z2 - z1
        dx = x2 - x1
        dy = y2 - y1
        # print(z1, x1, y1, z2, x2, y2)
        if dx <= 112:
            if dy <= 112:
                wd = int((112 - dx) / 2)
                hd = int((112 - dy) / 2)
                x_1, x_2 = max(x1 - wd, 0), min(x1 - wd + 112, lab_x)
                y_1, y_2 = max(y1 - hd, 0), min(y1 - hd + 112, lab_y)
                z_1, z_2 = z1, z1 + dz

                mask[z_1:z_2, x_1:x_2, y_1:y_2] = 0
                loss_mask[:, z_1:z_2, x_1:x_2, y_1:y_2] = 0
                # print(x_1, y_1, x_2, y_2)
            else:
                mask[z1:z1 + dz, x1:x1 + 112, y1:y1 + 112] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
                loss_mask[:, z1:z1 + dz, x1:x1 + 112, y1:y1 + 112] = 0
                x_1, x_2, y_1, y_2, z_1, z_2 = x1, x1 + 112, y1, y1 + 112, z1, z1 + dz
        else:
            mask[z1:z1 + dz, x1:x1 + 112, y1:y1 + 112] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
            loss_mask[:, z1:z1 + dz, x1:x1 + 112, y1:y1 + 112] = 0
            x_1, x_2, y_1, y_2, z_1, z_2 = x1, x1 + 112, y1, y1 + 112, z1, z1 + dz
    else:
        patch_z, patch_x, patch_y = int(lab_z * 1 / 2), int(lab_x * 1 / 2), int(lab_y * 1 / 2)  # 计算遮挡区域的大小为输入图像的1/2
        dz, dx, dy = np.random.randint(0, lab_z - patch_z), np.random.randint(0, lab_x - patch_x), np.random.randint(0,
                                                                                                                     lab_y - patch_y)
        mask[dz:dz + patch_z, dx:dx + patch_x, dy:dy + patch_y] = 0  # 将遮挡区域设置为0，即对应位置被遮挡。
        loss_mask[:, dz:dz + patch_z, dx:dx + patch_x, dy:dy + patch_y] = 0  # 在loss_mask中标记对应的遮挡区域。
        z_1 = np.random.randint(0, lab_z - patch_z)
        x_1 = np.random.randint(0, lab_x - patch_x)
        y_1 = np.random.randint(0, lab_y - patch_y)
        z_2, x_2, y_2 = z_1 + 112, x_1 + 112, y_1 + 112

    return mask.long(), loss_mask.long(), z_1, x_1, y_1, z_2, x_2, y_2


def context_mask(img, mask_ratio):
    # 获取当前时间作为种子
    seed = int(time.time())
    np.random.seed(seed)
    batch_size, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()

    nonzero_indices = torch.nonzero(img)
    top_left = torch.min(nonzero_indices, dim=0)[0]
    bottom_right = torch.max(nonzero_indices, dim=0)[0]
    # print("非零区域的左上坐标:", top_left.tolist())
    # print("非零区域的右下坐标:", bottom_right.tolist())
    x1, y1, z1 = top_left[1], top_left[2], top_left[3]

    x2, y2, z2 = bottom_right[1], bottom_right[2], bottom_right[3]
    x1 = x1.data.cpu()
    x2 = x2.data.cpu()
    y1 = y1.data.cpu()
    y2 = y2.data.cpu()
    z1 = z1.data.cpu()
    z2 = z2.data.cpu()
    w = np.random.randint(0, (x2 + x1) / 2)
    h = np.random.randint(0, (y2 + y1) / 2)
    z = np.random.randint(0, (z2 + z1) / 2)
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(
        img_z * mask_ratio)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    return mask.long(), loss_mask.long()


def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * 2 / 3), int(img_y * 2 / 3), int(img_z * 2 / 3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x / 3) + 1, int(patch_pixel_y / 3) + 1, int(
        patch_pixel_z / 3)
    size_x, size_y, size_z = int(img_x / 3), int(img_y / 3), int(img_z / 3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs * size_x, (xs + 1) * size_x - mask_size_x - 1)
                h = np.random.randint(ys * size_y, (ys + 1) * size_y - mask_size_y - 1)
                z = np.random.randint(zs * size_z, (zs + 1) * size_z - mask_size_z - 1)
                mask[w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0
                loss_mask[:, w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0
    return mask.long(), loss_mask.long()


def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length - 1)
    mask[:, :, z:z + z_length] = 0
    loss_mask[:, :, :, z:z + z_length] = 0
    return mask.long(), loss_mask.long()


def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss


# def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
#     CE = nn.CrossEntropyLoss(reduction='none')
#     img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
#     output_soft = F.softmax(output, dim=1)
#     image_weight, patch_weight = l_weight, u_weight
#     if unlab:
#         image_weight, patch_weight = u_weight, l_weight
#     patch_mask = 1 - mask

#     # 计算Dice损失
#     loss_dice = DICE(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
#     loss_dice += DICE(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight

#     # 计算交叉熵损失
#     loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
#     loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)

#     # 计算边界损失
#     # loss_boundary = boundary_loss(output, img_l.unsqueeze(1)) * image_weight
#     # loss_boundary += boundary_loss(output, patch_l.unsqueeze(1)) * patch_weight
#     loss_boundary = boundary_loss(output, img_l.unsqueeze(1),weight=0.1)* image_weight
#     loss_boundary += boundary_loss(output, patch_l.unsqueeze(1),weight=0.1)* image_weight
#     print("loss_bosss",loss_boundary)
#     # 返回综合损失
#     loss = (loss_dice + loss_ce + loss_boundary) / 3
#     return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha) / 2) * param1.data).add_(((1 - alpha) / 2) * param2.data)


@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data


class BBoxException(Exception):
    pass


def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))


def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

