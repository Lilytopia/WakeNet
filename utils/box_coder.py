import numpy as np
import torch


class BoxCoder(object):
    def __init__(self, weights1=(10., 10., 10., 10., 15.), weights2=(3., 3., 1.5, 1.5)):
        self.weights1 = weights1
        self.weights2 = weights2

    def encode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_thetas = ex_rois[:, 4]
        gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
        gt_widths = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
        gt_thetas = gt_rois[:, 4]
        wx, wy, ww, wh, wt = self.weights1
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets_dt = wt * (torch.tan(gt_thetas / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi))
        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh, targets_dt), dim=1
        )
        return targets

    def decode(self, boxes, deltas, mode='xywht'):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights
        thetas = boxes[:, :, 4]
        wx, wy, ww, wh, wt = self.weights1
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dw = deltas[:, :, 2] / ww
        dh = deltas[:, :, 3] / wh
        dt = deltas[:, :, 4] / wt
        pred_ctr_x = ctr_x if 'x' not in mode else ctr_x + dx * widths
        pred_ctr_y = ctr_y if 'y' not in mode else ctr_y + dy * heights
        pred_w = widths if 'w' not in mode else torch.exp(dw) * widths
        pred_h = heights if 'h' not in mode else torch.exp(dh) * heights
        pred_t = thetas if 't' not in mode else torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt) / np.pi * 180.0
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2,
            pred_t], dim=2
        )
        return pred_boxes

    def landmarkencode(self, ex_rois, gt_rois):
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
        ex_widths = torch.clamp(ex_widths, min=1)
        ex_heights = torch.clamp(ex_heights, min=1)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_theta = ex_rois[:, 4] * (ex_heights <= ex_widths) + (ex_rois[:, 4] + 90.0) * (ex_heights > ex_widths)
        ex_theta = ex_theta * (ex_theta >= 0) + (ex_theta + 180.0) * (ex_theta < 0)
        ex_theta = ex_theta * (ex_theta < 180.0) + (ex_theta - 180.0) * (ex_theta >= 180.0)
        gt_ctr_x = gt_rois[:, 0]
        gt_ctr_y = gt_rois[:, 1]
        mid1 = gt_rois[:, 2] * (gt_rois[:, 2] <= gt_rois[:, 3]) + gt_rois[:, 3] * (gt_rois[:, 2] > gt_rois[:, 3])
        mid2 = gt_rois[:, 3] * (gt_rois[:, 2] <= gt_rois[:, 3]) + gt_rois[:, 2] * (gt_rois[:, 2] > gt_rois[:, 3])
        gt_theta1 = mid1 * (abs(mid2 - mid1) <= 180.0) + mid2 * (abs(mid2 - mid1) > 180.0)
        gt_theta2 = mid2 * (abs(mid2 - mid1) <= 180.0) + mid1 * (abs(mid2 - mid1) > 180.0)
        gt_theta1 = gt_theta1 * (gt_theta1 >= 0) + (gt_theta1 + 180.0) * (gt_theta1 < 0)
        gt_theta1 = gt_theta1 * (gt_theta1 < 180.0) + (gt_theta1 - 180.0) * (gt_theta1 >= 180.0)
        gt_theta2 = gt_theta2 * (gt_theta2 >= 0) + (gt_theta2 + 180.0) * (gt_theta2 < 0)
        gt_theta2 = gt_theta2 * (gt_theta2 < 180.0) + (gt_theta2 - 180.0) * (gt_theta2 >= 180.0)
        delta_t1 = gt_theta1 - ex_theta
        delta_t2 = gt_theta2 - ex_theta
        delta_t1[torch.abs(delta_t1 - 90.0) <= 0.1] = delta_t1[torch.abs(delta_t1 - 90.0) <= 0.1] - 0.1
        delta_t2[torch.abs(delta_t2 - 90.0) <= 0.1] = delta_t2[torch.abs(delta_t2 - 90.0) <= 0.1] - 0.1
        delta_t1[torch.abs(delta_t1 + 90.0) <= 0.1] = delta_t1[torch.abs(delta_t1 + 90.0) <= 0.1] - 0.1
        delta_t2[torch.abs(delta_t2 + 90.0) <= 0.1] = delta_t2[torch.abs(delta_t2 + 90.0) <= 0.1] - 0.1
        wx, wy, wt1, wt2 = self.weights2
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dt1 = wt1 * (torch.tan(delta_t1 / 180.0 * np.pi))
        targets_dt2 = wt2 * (torch.tan(delta_t2 / 180.0 * np.pi))
        targets = torch.stack(
            (targets_dx, targets_dy, targets_dt1, targets_dt2), dim=1
        )
        return targets

    def landmarkdecode(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights
        theta = boxes[:, :, 4] * (heights <= widths) + (boxes[:, :, 4] + 90.0) * (heights > widths)
        theta = theta * (theta >= 0) + (theta + 180.0) * (theta < 0)
        theta = theta * (theta < 180.0) + (theta - 180.0) * (theta >= 180.0)
        wx, wy, wt1, wt2 = self.weights2
        dx = deltas[:, :, 0] / wx
        dy = deltas[:, :, 1] / wy
        dt1 = deltas[:, :, 2] / wt1
        dt2 = deltas[:, :, 3] / wt2
        pred_x = ctr_x + dx * widths
        pred_y = ctr_y + dy * heights
        pred_standard = torch.atan2(ctr_y-pred_y, ctr_x-pred_x) / np.pi * 180.0
        theta = theta * (abs(theta - pred_standard) <= 90.0) + theta * (abs(theta - pred_standard) > 270.0) + (
                theta - 180.0) * (abs(theta - pred_standard) > 90.0) * (abs(theta - pred_standard) <= 270.0)
        pred_delta1 = torch.atan(dt1) / np.pi * 180.0
        pred_delta2 = torch.atan(dt2) / np.pi * 180.0
        pred_delta1 = pred_delta1 * (abs(pred_delta1) <= 90.0) + (pred_delta1 - 180.0) * (abs(pred_delta1) > 90.0)
        pred_delta2 = pred_delta2 * (abs(pred_delta2) <= 90.0) + (pred_delta2 - 180.0) * (abs(pred_delta2) > 90.0)
        pred_theta1 = theta + pred_delta1
        pred_theta2 = theta + pred_delta2
        pred_boxes = torch.stack([
            pred_x,
            pred_y,
            pred_theta1,
            pred_theta2], dim=2
        )
        return pred_boxes

    def lossdecode(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        thetas = boxes[:, 4]
        wx, wy, ww, wh, wt = self.weights1
        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = deltas[:, 2] / ww
        dh = deltas[:, 3] / wh
        dt = deltas[:, 4] / wt
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_t = torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt) / np.pi * 180.0
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2,
            pred_t], dim=1
        )
        return pred_boxes

    def xywhdecode(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        widths = torch.clamp(widths, min=1)
        heights = torch.clamp(heights, min=1)
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        thetas = boxes[:, 4]
        wx, wy, ww, wh, wt = self.weights1
        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = deltas[:, 2] / ww
        dh = deltas[:, 3] / wh
        dt = deltas[:, 4] / wt
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_t = torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt)
        pred_xywhtboxes = torch.stack([
            pred_ctr_x,
            pred_ctr_y,
            pred_w,
            pred_h,
            pred_t], dim=1
        )
        return pred_xywhtboxes

    def lmr5pangle(self, anchors, deltas):
        wx, wy, ww, wh, wt = self.weights1
        thetas = anchors[:, 4]
        dt = deltas / wt
        pred_t = torch.atan(torch.tan(thetas / 180.0 * np.pi) + dt)
        targets_dt = wt * (torch.tan(pred_t - 0.5 * np.pi) - torch.tan(thetas / 180.0 * np.pi))
        return targets_dt
