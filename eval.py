from __future__ import print_function

import os
import cv2
import torch
import codecs
import shutil
import argparse
from tqdm import tqdm
from datasets import *
from models.WakeNet import WakeNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import sort_corners, is_image, hyp_parse
from utils.map import eval_mAP

DATASETS = {'SWIM': SWIMDataset}


def data_evaluate(model,
                  target_size,
                  test_path,
                  conf=0.01,
                  dataset=None):
    root_dir = 'datasets/evaluate'
    out_dir = os.path.join(root_dir, 'detection-results')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    ds = DATASETS[dataset]()

    with open(test_path, 'r') as f:
        if dataset == 'VOC':
            im_dir = test_path.replace('/ImageSets/Main/test.txt', '/JPEGImages')
            ims_list = [os.path.join(im_dir, x.strip('\n') + '.jpg') for x in f.readlines()]
        else:
            ims_list = [x.strip('\n') for x in f.readlines() if is_image(x.strip('\n'))]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_path in enumerate(tqdm(ims_list, desc=s)):
        im_name = os.path.split(im_path)[1]
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf=conf)
        nt += len(dets)
        out_file = os.path.join(out_dir, im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                    ds.return_class(dets[k, 0]), dets[k, 1],
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                )
        assert len(os.listdir(os.path.join(root_dir, 'ground-truth'))) != 0, 'No labels found in test/ground-truth!! '
    mAP = eval_mAP(root_dir, use_07_metric=True, thres=0.5)

    pf = '%20s' + '%10.4g' * 6
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    return 0, 0, mAP, 0


def evaluate(target_size,
             test_path,
             dataset,
             backbone=None,
             weight=None,
             model=None,
             hyps=None,
             conf=0.01):
    if model is None:
        model = WakeNet(backbone=backbone, hyps=hyps)
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)

            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    results = data_evaluate(model, target_size, test_path, conf, dataset)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='fca101', type=str)
    parser.add_argument('--weight', type=str, default='weights/best.pth')
    parser.add_argument('--target_size', dest='target_size', default=[768], type=int)
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--dataset', nargs='?', type=str, default='SWIM')
    parser.add_argument('--test_path', type=str, default='SWIM/test.txt')

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    evaluate(arg.target_size,
             arg.test_path,
             arg.dataset,
             arg.backbone,
             arg.weight,
             hyps=hyps)
