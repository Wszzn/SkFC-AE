# -*- coding: utf-8 -*-
import numpy as np
import torch
from datetime import datetime as dt

from torch import nn
from torch.nn import functional as F

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, best_epoch, best_iou, lie=None, lie_solver=None, gpe=None,
                     gpe_solver=None,
                     ffm=None, ffm_solver=None, decoder=None, decoder_solver=None, fcm=None, fcm_solver=None,
                     gie=None, gie_solver=None, mlp=None, mlp_solver=None):
    import torch
    from datetime import datetime as dt

    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))

    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch
    }

    # Save model and solver state dictionaries if they exist
    if lie is not None:
        checkpoint['lie_state_dict'] = lie.state_dict()
    if lie_solver is not None:
        checkpoint['lie_solver_state_dict'] = lie_solver.state_dict()

    if gpe is not None:
        checkpoint['gpe_state_dict'] = gpe.state_dict()
    if gpe_solver is not None:
        checkpoint['gpe_solver_state_dict'] = gpe_solver.state_dict()

    if ffm is not None:
        checkpoint['ffm_state_dict'] = ffm.state_dict()
    if ffm_solver is not None:
        checkpoint['ffm_solver_state_dict'] = ffm_solver.state_dict()

    if decoder is not None:
        checkpoint['decoder_state_dict'] = decoder.state_dict()
    if decoder_solver is not None:
        checkpoint['decoder_solver_state_dict'] = decoder_solver.state_dict()

    if fcm is not None:
        checkpoint['fcm_state_dict'] = fcm.state_dict()
    if fcm_solver is not None:
        checkpoint['fcm_solver_state_dict'] = fcm_solver.state_dict()

    if gie is not None:
        checkpoint['gie_state_dict'] = gie.state_dict()
    if gie_solver is not None:
        checkpoint['gie_solver_state_dict'] = gie_solver.state_dict()

    if mlp is not None:
        checkpoint['mlp_state_dict'] = mlp.state_dict()
    if mlp_solver is not None:
        checkpoint['mlp_solver_state_dict'] = mlp_solver.state_dict()

    # Save the checkpoint to file
    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def MSFCEL(pred_value, ground_truth):
    """
        Mean Squared False Cross-Entropy Loss function
    @param ground_truth: ground truth volumes
    @param pred_value: generated_volumes
    @return: loss value in float
    """
    bce_loss = torch.nn.BCELoss()
    unoccupied_voxels = torch.eq(ground_truth, 0)
    occupied_voxels = torch.eq(ground_truth, 1)
    Fpos_ce = bce_loss(pred_value[unoccupied_voxels], ground_truth[unoccupied_voxels])
    Fneg_ce = bce_loss(pred_value[occupied_voxels], ground_truth[occupied_voxels])
    loss = (torch.pow(Fpos_ce, 2) + torch.pow(Fneg_ce, 2))*100
    # loss = F.binary_cross_entropy(pred_value, ground_truth, reduction="mean")*1000
    return loss

def KL(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def MSE(pred_value, ground_truth):
    # loss1 = 0.1 * torch.nn.MSELoss(reduction='sum')(pred_value, ground_truth)
    projection_x = torch.max(pred_value, dim=1, keepdim=False)[0]
    projection_y = torch.max(pred_value, dim=2, keepdim=False)[0]
    projection_z = torch.max(pred_value, dim=3, keepdim=False)[0]

    projection_x_hat = torch.max(ground_truth, dim=1, keepdim=False)[0]
    projection_y_hat = torch.max(ground_truth, dim=2, keepdim=False)[0]
    projection_z_hat = torch.max(ground_truth, dim=3, keepdim=False)[0]

    loss2 = torch.nn.MSELoss(reduction='sum')(projection_x, projection_x_hat) + torch.nn.MSELoss(
        reduction='sum')(projection_y, projection_y_hat) + torch.nn.MSELoss(reduction='sum')(projection_z,
                                                                                                   projection_z_hat)
    return 0.1*loss2

def DiceLoss(predict, target, smooth=1, p=2, reduction='sum'):
    """based on https://github.com/hubutui/DiceLoss-PyTorch"""
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = torch.sigmoid(predict)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    loss = (1 - num / den)*100

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))

def Total_Loss(pred_value, ground_truth, mu, logvar):
    # l1 = MSFCEL(pred_value, ground_truth)
    l1 = MSFCEL(pred_value, ground_truth)
    l2 = KL(mu, logvar)
    l3 = MSE(pred_value, ground_truth)
    return l1+l2+l3

def MSE_MSFCEL(pred_value, ground_truth):
    l1 = MSE(pred_value, ground_truth)
    l2 = MSFCEL(pred_value, ground_truth)
    return l1+l2
