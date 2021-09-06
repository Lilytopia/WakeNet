import os
import cv2
import numpy as np
import shutil
import math
from tqdm import tqdm


def rbox_2_quad(rbox):
    quads = np.zeros((8), dtype=np.float32)
    x = rbox[0]
    y = rbox[1]
    w = rbox[2]
    h = rbox[3]
    theta = rbox[4]
    quads = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))
    return quads


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j * 2] = corners[idx * 2]
            sorted[i, j * 2 + 1] = corners[idx * 2 + 1]
    return sorted


def convert_swim_gt(gt_path, dst_path, eval_difficult=False):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    with open(gt_path, 'r') as f:
        files = [x.strip('\n').replace('.jpg', '.xml').replace('JPEGImages', 'Annotations') for x in f.readlines()]
    gts = [os.path.split(x)[1] for x in files]
    dst_gt = [os.path.join(dst_path, x.replace('.xml', '.txt')) for x in gts]
    print('gt generating...')
    for i, filename in enumerate(tqdm(files)):
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('<object>')
            info = objects.pop(0)
            nt = 0
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s' % filename
                diffculty = obj[obj.find('<difficult>') + 11: obj.find('</difficult>')]
                nt += 1
                cx = round(eval(obj[obj.find('<cx>') + 4: obj.find('</cx>')]))
                cy = round(eval(obj[obj.find('<cy>') + 4: obj.find('</cy>')]))
                w = round(eval(obj[obj.find('<w>') + 3: obj.find('</w>')]))
                h = round(eval(obj[obj.find('<h>') + 3: obj.find('</h>')]))
                a = eval(obj[obj.find('<angle>') + 7: obj.find('</angle>')]) / math.pi * 180
                pt = sort_corners(rbox_2_quad([cx, cy, w, h, a]))[0]
                name = 'wake'
                with open(dst_gt[i], 'a') as fd:
                    if eval_difficult:
                        fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            name, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]
                        ))
                    else:
                        if diffculty == '0':
                            fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                                name, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]
                            ))
                        elif diffculty == '1':
                            fd.write('{} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} difficult\n'.format(
                                name, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]
                            ))
                        else:
                            raise RuntimeError('Wrong difficulty.')
            if nt == 0:
                open(dst_gt[i], 'w').close()


if __name__ == "__main__":
    gt_path = '/SWIM/test.txt'
    dst_path = '/datasets/evaluate/ground-truth'
    eval_difficult = False
    convert_swim_gt(gt_path, dst_path, eval_difficult)
