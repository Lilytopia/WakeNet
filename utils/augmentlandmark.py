import random
import cv2
import numpy as np
import imgaug.augmenters as iaa
from utils.bbox import quad_2_rbox, rbox_2_quad, points_2_thetas, thetas_2_points


class HSV(object):
    def __init__(self, saturation=0, brightness=0, p=0.):
        self.saturation = saturation
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels, landmarks


class HSV_pos(object):
    def __init__(self, saturation=0, brightness=0, p=0.):
        self.saturation = saturation
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(0, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels, landmarks


class Blur(object):
    def __init__(self, sigma=0, p=0.):
        self.sigma = sigma
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0, self.sigma))
            img = blur_aug.augment_image(img)
        return img, labels, landmarks


class Grayscale(object):
    def __init__(self, grayscale=0., p=0.):
        self.alpha = random.uniform(grayscale, 1.0)
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, labels, landmarks


class Gamma(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            gm = random.uniform(1 - self.intensity, 1 + self.intensity)
            img = np.uint8(np.power(img / float(np.max(img)), gm) * np.max(img))
        return img, labels, landmarks


class Noise(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, labels, landmarks


class Sharpen(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity, 1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, labels, landmarks


class Contrast(object):
    def __init__(self, intensity=0, p=0.):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            contrast_aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img = contrast_aug.augment_image(img)
        return img, labels, landmarks


class HorizontalFlip(object):
    def __init__(self, p=0.):
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            img = np.fliplr(img)
            if mode == 'cxywha':
                labels[:, 1] = img.shape[1] - labels[:, 1]
                labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                labels[:, [0, 2, 4, 6]] = img.shape[1] - labels[:, [0, 2, 4, 6]]
            if mode == 'xywha':
                labels[:, 0] = img.shape[1] - labels[:, 0]
                labels[:, -1] = -labels[:, -1]
            landmarks[:, 0] = img.shape[1] - landmarks[:, 0]
            landmarks[:, [-2, -1]] = -landmarks[:, [-2, -1]]
        return img, labels, landmarks


class VerticalFlip(object):
    def __init__(self, p=0.):
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            img = np.flipud(img)
            if mode == 'cxywha':
                labels[:, 2] = img.shape[0] - labels[:, 2]
                labels[:, 5] = -labels[:, 5]
            if mode == 'xyxyxyxy':
                labels[:, [1, 3, 5, 7]] = img.shape[0] - labels[:, [1, 3, 5, 7]]
            if mode == 'xywha':
                labels[:, 1] = img.shape[0] - labels[:, 1]
                labels[:, -1] = -labels[:, -1]
            landmarks[:, 1] = img.shape[0] - landmarks[:, 1]
            landmarks[:, -1] = (180.0 - landmarks[:, -1]) * (landmarks[:, -1] > 0) + (-180.0 - landmarks[:, -1]) * (
                    landmarks[:, -1] < 0)
            landmarks[:, -2] = (180.0 - landmarks[:, -2]) * (landmarks[:, -2] > 0) + (-180.0 - landmarks[:, -2]) * (
                    landmarks[:, -2] < 0)
        return img, labels, landmarks


class Affine(object):
    def __init__(self, degree=0., translate=0., scale=0., shear=0., p=0.):
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img, labels, landmarks, mode=None):
        if random.random() < self.p:
            if mode == 'xywha':
                labels = rbox_2_quad(labels, mode='xywha')
                landmarks = thetas_2_points(landmarks)
                img, labels, landmarks = random_affine(img, labels, landmarks,
                                                       degree=self.degree, translate=self.translate,
                                                       scale=self.scale, shear=self.shear)
                labels = quad_2_rbox(labels, mode='xywha')
                landmarks = points_2_thetas(landmarks)

            else:
                landmarks = thetas_2_points(landmarks)
                img, labels, landmarks = random_affine(img, labels, landmarks,
                                                       degree=self.degree, translate=self.translate,
                                                       scale=self.scale, shear=self.shear)
                landmarks = points_2_thetas(landmarks)
        return img, labels, landmarks


class Augment(object):
    def __init__(self, augmentations, probs=1, box_mode=None):
        self.augmentations = augmentations
        self.probs = probs
        self.mode = box_mode

    def __call__(self, img, labels, landmarks):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                img, labels, landmarks = augmentation(img, labels, landmarks, self.mode)

        return img, labels, landmarks


def random_affine(img, targets1=None, targets2=None, degree=10, translate=.1, scale=.1, shear=10):
    if targets1 is None:
        targets1 = []
    if targets2 is None:
        targets2 = []
    border = 0
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border
    M = T @ R
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))
    targets1[:, [0, 2, 4, 6]] = targets1[:, [0, 2, 4, 6]] * M[0, 0] + targets1[:, [1, 3, 5, 7]] * M[0, 1] + M[0, 2]
    targets1[:, [1, 3, 5, 7]] = targets1[:, [0, 2, 4, 6]] * M[1, 0] + targets1[:, [1, 3, 5, 7]] * M[1, 1] + M[1, 2]
    targets2[:, [0, 2, 4]] = targets2[:, [0, 2, 4]] * M[0, 0] + targets2[:, [1, 3, 5]] * M[0, 1] + M[0, 2]
    targets2[:, [1, 3, 5]] = targets2[:, [0, 2, 4]] * M[1, 0] + targets2[:, [1, 3, 5]] * M[1, 1] + M[1, 2]
    for x in range(0, 8, 2):
        targets1[:, x] = targets1[:, x].clip(0, width)
    for y in range(1, 8, 2):
        targets1[:, y] = targets1[:, y].clip(0, height)
    return imw, targets1, targets2
