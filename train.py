from __future__ import print_function

import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
from tqdm import tqdm

from models.WakeNet import WakeNet
from eval import evaluate
from datasets import *
from utils.utils import *
from torch_warmup_lr import WarmupLR

mixed_precision = True
try:
    from apex import amp
except:
    print('Fail to speed up training via apex. \n')
    mixed_precision = False

DATASETS = {'SWIM': SWIMDataset}


def train_model(args, hyps):
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    results_file = 'result.txt'
    weight = 'weights' + os.sep + 'last.pth' if args.resume or args.load else args.weight
    last = 'weights' + os.sep + 'last.pth'
    best = 'weights' + os.sep + 'best.pth'
    start_epoch = 0
    best_fitness = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        os.remove(f)

    if args.multi_scale:
        scales = args.training_size + 32 * np.array([x for x in range(-1, 5)])

        print('Using multi-scale %g - %g' % (scales[0], scales[-1]))
    else:
        scales = args.training_size

    assert args.dataset in DATASETS.keys(), 'Not supported dataset!'
    ds = DATASETS[args.dataset](dataset=args.train_path, augment=args.augment)
    collater = Collater(scales=scales, keep_ratio=True, multiple=32)
    loader = data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collater,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    init_seeds()
    model = WakeNet(backbone=args.backbone, hyps=hyps)

    optimizer = optim.Adam(model.parameters(), lr=hyps['lr0'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    scheduler = WarmupLR(scheduler, init_lr=hyps['warmup_lr'], num_warmup=hyps['warm_epoch'], warmup_strategy='cos')
    scheduler.last_epoch = start_epoch - 1

    if weight.endswith('.pth'):
        chkpt = torch.load(weight)
        print('Weight loaded.')

        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
            print('Model loaded.')
        else:
            model.load_state_dict(chkpt)

        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            print('Optimizer loaded.')

        if 'training_results' in chkpt.keys() and chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])
            print('Result loaded.')
        if args.resume and 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1
        del chkpt

    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    model_info(model, report='summary')
    results = (0, 0, 0, 0)

    for epoch in range(start_epoch, epochs):
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'cls', 'reg_box', 'reg_ldm', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(loader), total=len(loader))
        mloss = torch.zeros(3).cuda()
        for i, (ni, batch) in enumerate(pbar):
            model.train()
            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()
            optimizer.zero_grad()
            ims, gt_boxes, gt_landmarks = batch['image'], batch['boxes'], batch['landmarks']
            if torch.cuda.is_available():
                ims, gt_boxes, gt_landmarks = ims.cuda(), gt_boxes.cuda(), gt_landmarks.cuda()
            losses = model(ims, gt_boxes, gt_landmarks, process=epoch / epochs)
            loss_cls, loss_reg1, loss_reg2 = losses['loss_cls'].mean(), losses['loss_reg1'].mean(), losses[
                'loss_reg2'].mean()
            loss = loss_cls + loss_reg1 * (hyps['lambda1']) + loss_reg2 * (hyps['lambda2'])
            if not torch.isfinite(loss):
                import ipdb
                ipdb.set_trace()
                print('WARNING: non-finite loss, ending training')
                break
            if bool(loss == 0):
                continue
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            loss_items = torch.stack([loss_cls, loss_reg1, loss_reg2], 0).detach()
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss,
                                               mloss[0] + mloss[1] * hyps['lambda1'] + mloss[2] * hyps['lambda2'],
                                               gt_boxes.shape[1], min(ims.shape[2:]))
            pbar.set_description(s)
        scheduler.step()
        final_epoch = epoch + 1 == epochs

        if hyps['test_interval'] != -1 and epoch % hyps['test_interval'] == 0 and epoch >= 20:
            if torch.cuda.device_count() > 1:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.eval_path,
                                   dataset=args.dataset,
                                   model=model.module,
                                   hyps=hyps,
                                   conf=0.01)
            else:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.eval_path,
                                   dataset=args.dataset,
                                   model=model,
                                   hyps=hyps,
                                   conf=0.01)

        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 4 % results + '\n')

        fitness = results[-2]
        if fitness > best_fitness:
            best_fitness = fitness

        with open(results_file, 'r') as f:
            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'training_results': f.read(),
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}
        torch.save(chkpt, last)

        if best_fitness == fitness:
            torch.save(chkpt, best)

        if (epoch % hyps['save_interval'] == 0 and epoch > 10) or final_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(chkpt, './weights/deploy%g.pth' % epoch)
            else:
                torch.save(chkpt, './weights/deploy%g.pth' % epoch)

    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--backbone', type=str, default='fca101_trick')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--dataset', type=str, default='SWIM')
    parser.add_argument('--train_path', type=str, default='SWIM/train.txt')
    parser.add_argument('--eval_path', type=str, default='SWIM/test.txt')
    parser.add_argument('--training_size', type=int, default=768)
    parser.add_argument('--resume', type=bool, default=False, help='resume training from last.pth')
    parser.add_argument('--load', type=bool, default=False, help='load training from last.pth')
    parser.add_argument('--augment', type=bool, default=True, help='data augment')
    parser.add_argument('--target_size', type=int, default=[768])

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    print(arg)
    print(hyps)

    train_model(arg, hyps)
