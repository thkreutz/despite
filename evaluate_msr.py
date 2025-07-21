from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import argparse
from src.utils import utils

from src.scheduler import WarmupMultiStepLR
from src.dataset.msr import MSRAction3D
from src.models import classifier
from src.models import model_loader
from src.models import SPITE

import wandb

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for clip, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)
        output = model(clip)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        if lr_scheduler:
            lr_scheduler.step()
        sys.stdout.flush()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, target, video_idx in metric_logger.log_every(data_loader, 100, header):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += (v==label)
    class_acc = [c/float(s) for c, s in zip(class_correct, class_count)]

    print(' * Video Acc@1 %f'%total_acc)
    print(' * Class Acc@1 %s'%str(class_acc))

    return total_acc

if __name__ == "__main__":
    
    
    ##### MODEL INIT MY STUFF
    parser = argparse.ArgumentParser(description='MSRAction3D CLS Finetuning')
    parser.add_argument('--model_type', default='SPITE', type=str, help='which model to use')
    parser.add_argument('--pretrained_path', default="no", type=str, help='which model to use, no input means train from scratch')
    parser.add_argument('--dataset', default="v1", type=str, help='pretrained on which dataset')
    parser.add_argument('--data-path', default='../MSRAction3D/data', type=str, help='dataset')
    parser.add_argument('--n_frames', default=24, type=int, help='number of frames')
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size')
    parser.add_argument('--epochs', default=35, type=int, help='Epochs')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed dim')
    parser.add_argument('--num_points', default=2048, type=int, help='number of points')
    parser.add_argument('--freeze', default=1, type=int, help='freeze weights and add projection layer')
    parser.add_argument('--projection', default="linear", type=str, help='which projection layer to use')
    parser.add_argument('--wandb', default=0, type=int, help='with wandb logging or no')
    parser.add_argument('--modality', default="pc", type=str, help='default modality is pointcloud')
    input_args = parser.parse_args()
    
    print(f"Running with configuration:")
    print(f"Model Type      : {input_args.model_type}")
    print(f"Pretrained Path : {input_args.pretrained_path if input_args.pretrained_path else 'Training from scratch'}")
    print(f"Number of Frames: {input_args.n_frames}") # valid = 10, 16, 24
    print(f"Batch Size      : {input_args.batch_size}")
    print(f"Epochs          : {input_args.epochs}")
    print(f"Embed Dim       : {input_args.embed_dim}")
    print(f"Number of Points: {input_args.num_points}")
    print(f"Freeze          : {input_args.freeze}")
    print(f"Probing         : {input_args.projection}")
    print(f"Wandb logging   : {input_args.wandb}")

    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')


    # Data loading code
    print("Loading data")

    st = time.time()
    dataset = MSRAction3D(
            root=input_args.data_path,
            frames_per_clip=input_args.n_frames,
            step_between_clips=1,
            num_points=input_args.num_points,
            train=True
    )

    dataset_test = MSRAction3D(
            root=input_args.data_path,
            frames_per_clip=input_args.n_frames,
            step_between_clips=1,
            num_points=input_args.num_points,
            train=False
    )

    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=input_args.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=input_args.batch_size, num_workers=1, pin_memory=False)

    ##### MODELS - always the same, just init.
    model_type =  input_args.model_type
    load_pretrained = input_args.pretrained_path
    embed_dim = input_args.embed_dim
    num_joints = 24
    n_feats = 3
    
    skeleton = model_loader.load_skeleton_encoder(embed_dim, num_joints, n_feats, device="cuda") if "S" in model_type else None
    imu = model_loader.load_imu_encoder(embed_dim, device="cuda") if "I" in model_type else None
    pc = model_loader.load_pst_transformer(embed_dim, device="cuda") if "P" in model_type else None
    skeleton_gen = model_loader.load_skeleton_generator(embed_dim, num_joints, n_feats, device="cuda") if "Gen" in model_type else None

    binder = SPITE.instantiate_binder_class_from_name(model_type, imu, pc, skeleton)

    # load state dict
    if input_args.pretrained_path != "no":
        binder.load_state_dict(torch.load(input_args.pretrained_path))

    ### Select model to use
    if input_args.modality == "pc":
        backbone = binder.pointcloud_encoder

    elif input_args.modality == "imu":
        backbone = binder.imu_encoder

    elif input_args.modality == "skeleton":
        backbone = binder.skeleton_encoder

    n_classes = dataset.num_classes
    ### Warp classifier
    if input_args.modality == "pc":
        classifier_model = classifier.ClassifierWrapperH(backbone, n_classes, freeze_backbone=input_args.freeze, probing = input_args.projection).to("cuda")
    else:
        classifier_model = classifier.ClassifierWrapperN(backbone, input_args.embed_dim, n_classes, freeze_backbone=input_args.freeze, probing = input_args.projection).to("cuda")
    
    #### CLS training parameters => Based on PST-Transformer on MSR-Action3D
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    lr_warmup_epochs = 10
    lr_milestones = [20, 30]
    lr_gamma = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    optimizer = torch.optim.SGD(classifier_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    with_wandb = input_args.wandb
    if with_wandb:
        # Initialize WandB
        wandb.init(project="SPITE_EVAL_MSR", config={
            "num_points" : input_args.num_points,
            "dataset" : input_args.dataset,
            "batch_size" : input_args.batch_size,
            "n_epochs" : input_args.epochs,
            "num_frames" : input_args.n_frames,
            "embed_dim" : input_args.embed_dim,
            "pretrained" : input_args.pretrained_path,
            "model" : input_args.model_type,
            "freeze" : input_args.freeze,
            "projection" : input_args.projection
        })

    print("Start training")
    start_time = time.time()
    acc = 0
    for epoch in range(0, input_args.epochs):
        train_one_epoch(classifier_model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, 20)
        acc_test = evaluate(classifier_model, criterion, data_loader_test, device=device)
        acc = max(acc, acc_test)

        if with_wandb:
            wandb.log({"epoch": epoch,
                        "acc_test" : acc_test,
                        "acc_max" : acc
                        }
                    )

        if acc_test == acc: # if max accuracy has changed to this one, save the model so we end up with best cls model.
            if with_wandb:
                torch.save(classifier_model.state_dict(), os.path.join(wandb.run.dir, "cls_model.pth"))    

        #if args.output_dir:
        #    checkpoint = {
        #        'model': model_without_ddp.state_dict(),
        #        'optimizer': optimizer.state_dict(),
        #        #'lr_scheduler': lr_scheduler.state_dict(),
        #        'epoch': epoch,
        #        'args': args}
            #utils.save_on_master(
            #    checkpoint,
            #    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            #utils.save_on_master(
            #    checkpoint,
            #    os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Accuracy {}'.format(acc))
