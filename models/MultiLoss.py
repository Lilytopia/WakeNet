import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bbox import bbox_overlaps, min_area_square, xy2wh
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps


def xyxy2xywh_a(query_boxes):
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes


class MultiLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func='lmr5p'):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron1 = smooth_l1_loss
        elif func == 'mse':
            self.criteron1 = F.mse_loss
        elif func == 'lmr5p':
            self.criteron1 = lmr5p
        elif func == 'balanced':
            self.criteron1 = balanced_l1_loss
        self.criteron2 = balanced_l1_loss

    def forward(self, classifications, regressions, anchors, annotations, landmarks=None, gt_linestrips=None,
                iou_thres=0.5):
        cls_losses = []
        reg_losses1 = []
        reg_losses2 = []
        batch_size = classifications.shape[0]
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            landmark = landmarks[j, :, :]
            bbox_annotation = annotations[j, :, :]
            no_background_mask = bbox_annotation[:, -1] != -1
            bbox_annotation = bbox_annotation[no_background_mask]
            if gt_linestrips is not None:
                landmark_annotation = gt_linestrips[j, :, :]
                landmark_annotation = landmark_annotation[no_background_mask]
            if bbox_annotation.shape[0] == 0:
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses1.append(torch.tensor(0).float().cuda())
                reg_losses2.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            ious = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                indicator.cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(ious):
                ious = torch.Tensor(ious).cuda()
            iou_max, iou_argmax = torch.max(ious, dim=1)
            positive_indices = torch.ge(iou_max, iou_thres)
            max_gt, argmax_gt = ious.max(0)
            if (max_gt < iou_thres).any():
                positive_indices[argmax_gt[max_gt < iou_thres]] = 1
            num_positive_anchors = positive_indices.sum()
            cls_targets = (torch.ones(classification.shape) * -1).cuda()
            cls_targets[torch.lt(iou_max, iou_thres - 0.1), :] = 0
            assigned_annotations = bbox_annotation[iou_argmax, :]
            if gt_linestrips is not None:
                assigned_landmarks = landmark_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1
            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(cls_targets * torch.log(classification + 1e-6) + (1.0 - cls_targets) * torch.log(
                1.0 - classification + 1e-6))
            cls_loss = focal_weight * bin_cross_entropy
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_targets1 = self.box_coder.encode(all_rois, gt_boxes)

                if self.criteron1 == lmr5p:
                    reg_loss1 = self.criteron1(regression[positive_indices, :], reg_targets1, all_rois, self.box_coder)
                else:
                    reg_loss1 = self.criteron1(regression[positive_indices, :], reg_targets1)
                reg_losses1.append(reg_loss1)
                if not torch.isfinite(reg_loss1):
                    import ipdb
                    ipdb.set_trace()

                if landmarks is not None and gt_linestrips is not None:
                    gt_landmarks = assigned_landmarks[positive_indices, :]
                    reg_targets2 = self.box_coder.landmarkencode(all_rois, gt_landmarks)
                    reg_loss2 = self.criteron2(landmark[positive_indices, :], reg_targets2)
                    reg_losses2.append(reg_loss2)
                    if not torch.isfinite(reg_loss2):
                        import ipdb
                        ipdb.set_trace()
                else:
                    reg_losses2.append(torch.tensor(0).float().cuda())
            else:
                reg_losses1.append(torch.tensor(0).float().cuda())
                reg_losses2.append(torch.tensor(0).float().cuda())

        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg1 = torch.stack(reg_losses1).mean(dim=0, keepdim=True)
        loss_reg2 = torch.stack(reg_losses2).mean(dim=0, keepdim=True)
        return loss_cls, loss_reg1, loss_reg2


def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight=None):
    diff = torch.abs(inputs - targets)
    if weight is None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1, 5)
    if size_average:
        return loss.mean()
    return loss.sum()


def lmr5p(inputs,
          targets,
          anchors,
          boxcoder,
          beta=1. / 9,
          size_average=True):
    assert inputs.shape[-1] == 5
    assert targets.shape[-1] == 5

    x2, x1 = inputs[:, 0], targets[:, 0]
    y2, y1 = inputs[:, 1], targets[:, 1]
    w2, w1 = inputs[:, 2], targets[:, 2]
    h2, h1 = inputs[:, 3], targets[:, 3]
    theta2, theta1 = inputs[:, 4], targets[:, 4]
    theta22 = boxcoder.lmr5pangle(anchors, theta2)

    diff1 = torch.abs(x1 - x2)
    diff2 = torch.abs(y1 - y2)
    diff3 = torch.abs(w1 - w2)
    diff4 = torch.abs(h1 - h2)
    diff5 = torch.abs(w1 - h2)
    diff6 = torch.abs(h1 - w2)
    diff7 = torch.abs(theta1 - theta2)
    diff8 = torch.abs(theta1 - theta22)
    loss1 = torch.where(diff1 < beta, 0.5 * diff1 ** 2 / beta, diff1 - 0.5 * beta)
    loss2 = torch.where(diff2 < beta, 0.5 * diff2 ** 2 / beta, diff2 - 0.5 * beta)
    loss3 = torch.where(diff3 < beta, 0.5 * diff3 ** 2 / beta, diff3 - 0.5 * beta)
    loss4 = torch.where(diff4 < beta, 0.5 * diff4 ** 2 / beta, diff4 - 0.5 * beta)
    loss5 = torch.where(diff5 < beta, 0.5 * diff5 ** 2 / beta, diff5 - 0.5 * beta)
    loss6 = torch.where(diff6 < beta, 0.5 * diff6 ** 2 / beta, diff6 - 0.5 * beta)
    loss7 = torch.where(diff7 < beta, 0.5 * diff7 ** 2 / beta, diff7 - 0.5 * beta)
    loss8 = torch.where(diff8 < beta, 0.5 * diff8 ** 2 / beta, diff8 - 0.5 * beta)

    loss = torch.min(
        loss1 + loss2 + loss3 + loss4 + loss7,
        loss1 + loss2 + loss5 + loss6 + loss8
    )

    if size_average:
        loss = loss / 5.0
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=0.5,
                     alpha=0.5,
                     gamma=0.9,
                     size_average=True):
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(diff < beta,
                       alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
                       gamma * diff + gamma / b - alpha * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
