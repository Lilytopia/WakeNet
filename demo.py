from __future__ import print_function
import os
import cv2
import time
import torch
import argparse
import numpy as np
from datasets import *
from models.WakeNet import WakeNet
from utils.detect import im_detect, im_detect_new
from utils.bbox import rbox_2_quad, thetas_2_5points
from utils.utils import is_image, draw_caption, hyp_parse

DATASETS = {'SWIM': SWIMDataset}


def demo(args):
    hyps = hyp_parse(args.hyp)
    ds = DATASETS[args.dataset]()
    model = WakeNet(backbone=args.backbone, hyps=hyps)
    new_wakenet = True

    if args.weight.endswith('.pth'):
        chkpt = torch.load(args.weight)
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))
    model.eval()

    t0 = time.time()
    ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]
    for idx, im_name in enumerate(ims_list):
        s = ''
        t = time.time()
        im_path = os.path.join(args.ims_dir, im_name)
        s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)
        src = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        if new_wakenet is not True:
            cls_dets = im_detect(model, im, target_sizes=args.target_size, use_gpu=True, conf=args.test_conf)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                                  thickness=2)
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, color=(0, 255, 0), thickness=2)
                    put_label = True
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.7
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                        cv2.rectangle(src, c1, c2, [0, 255, 0], -1)
                        cv2.putText(src, label, (c1[0], c1[1] - 5), font, fontScale, [0, 0, 0], thickness=thickness,
                                    lineType=cv2.LINE_AA)

        else:
            cls_dets = im_detect_new(model, im, target_sizes=args.target_size, use_gpu=True, conf=args.test_conf)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                ldm = cls_dets[j, 2:6]
                bbox = cls_dets[j, 6:]
                pts_bbx = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                cv2.drawContours(src, pts_bbx, 0, color=(0, 255, 0), thickness=2)
                pts_ldm = np.array([thetas_2_5points(ldm)], dtype=np.int32)
                cv2.line(src, (pts_ldm[:, :, 0], pts_ldm[:, :, 1]), (pts_ldm[:, :, 2], pts_ldm[:, :, 3]), (255, 0, 0),
                         5)
                cv2.line(src, (pts_ldm[:, :, 0], pts_ldm[:, :, 1]), (pts_ldm[:, :, 4], pts_ldm[:, :, 5]), (255, 0, 0),
                         5)
                cv2.circle(src, (pts_ldm[:, :, 0], pts_ldm[:, :, 1]), 5, (0, 255, 255), -1)
                put_label = False
                if put_label:
                    label = ds.return_class(cls) + str(' %.2f' % scores)
                    fontScale = 0.7
                    font = cv2.FONT_HERSHEY_COMPLEX
                    thickness = 1
                    t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                    c1 = tuple(bbox[:2].astype('int'))
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                    cv2.rectangle(src, c1, c2, [0, 255, 0], -1)
                    cv2.putText(src, label, (c1[0], c1[1] - 5), font, fontScale, [0, 0, 0], thickness=thickness,
                                lineType=cv2.LINE_AA)

        print('%sDone. (%.3fs) %d objs' % (s, time.time() - t, len(cls_dets)))

        out_path = os.path.join('outputs', os.path.split(im_path)[1])
        cv2.imwrite(out_path, src)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='fca101')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/best.pth')
    parser.add_argument('--dataset', type=str, default='SWIM')
    parser.add_argument('--ims_dir', type=str, default='test')
    parser.add_argument('--test_conf', type=float, default=0.5)
    parser.add_argument('--target_size', type=int, default=[768])
    demo(parser.parse_args())
