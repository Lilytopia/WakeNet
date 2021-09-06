import torch
from utils.nms.cpu_nms import cpu_nms, cpu_soft_nms


def nms(dets, thresh, use_gpu=False):
    if dets.shape[0] == 0:
        return []
    if dets.shape[1] == 5:
        raise NotImplementedError
    elif dets.shape[1] == 6:
        if torch.is_tensor(dets):
            dets = dets.cpu().detach().numpy()
        return cpu_nms(dets, thresh)
    else:
        raise NotImplementedError
