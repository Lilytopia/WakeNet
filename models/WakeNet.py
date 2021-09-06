import torch
import torch.nn as nn
import numpy as np
from models.FcaNet import fcanet34, fcanet50, fcanet101, fcanet152
from models.FPAN import FPAN, LastLevelP6P7
from models.MultiHeads import CLSHead, REGHead, LDMHead
from models.Anchors import Anchors
from models.MultiLoss import MultiLoss
from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes, clip_landmarks


class WakeNet(nn.Module):
    def __init__(self, backbone='fca101', hyps=None):
        super(WakeNet, self).__init__()
        self.num_classes = int(hyps['num_classes']) + 1
        self.anchor_generator = Anchors(
            ratios=np.array([0.2, 0.5, 1.0, 2.0, 5.0]),
        )
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        self.fpan = FPAN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256)
        )
        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5
        )
        self.ldm_head0 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p3')
        self.ldm_head1 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p4')
        self.ldm_head2 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p5')
        self.ldm_head3 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p6')
        self.ldm_head4 = LDMHead(in_channels=256, feat_channels=256, num_anchors=self.num_anchors, num_landmarks=4,
                                 level='p7')
        self.loss = MultiLoss(func='lmr5p')
        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'fca34':
            self.backbone = fcanet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca50':
            self.backbone = fcanet50(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca101':
            self.backbone = fcanet101(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca152':
            self.backbone = fcanet152(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        else:
            raise NotImplementedError

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c3, c4, c5]

    def forward(self, imgs, gt_boxes=None, gt_landmarks=None, test_conf=None, process=None):
        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(imgs)
        anchors_list.append(original_anchors)
        features = self.fpan(self.ims_2_features(imgs))
        cls_score = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        land_pred = torch.cat([self.ldm_head0(features[0]), self.ldm_head1(features[1]), self.ldm_head2(features[2]),
                               self.ldm_head3(features[3]), self.ldm_head4(features[4])], dim=1)
        if self.training:
            losses = dict()
            losses['loss_cls'], losses['loss_reg1'], losses['loss_reg2'] = self.loss(cls_score, bbox_pred,
                                                                                     anchors_list[-1], gt_boxes,
                                                                                     landmarks=land_pred,
                                                                                     gt_linestrips=gt_landmarks,
                                                                                     iou_thres=0.5)
            return losses
        else:
            return self.decoder(imgs, anchors_list[-1], cls_score, bbox_pred, land_pred, test_conf=test_conf)

    def decoder(self, imgs, anchors, cls_score, bbox_pred, landmark_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, imgs)
        landmarks = self.box_coder.landmarkdecode(anchors, landmark_pred)
        landmarks = clip_landmarks(landmarks, imgs)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5), torch.zeros(1, 4)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        landmarks = landmarks[:, keep, :]
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        output_landmarks = landmarks[0, anchors_nms_idx, :]
        return [nms_scores, nms_class, output_boxes, output_landmarks]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
