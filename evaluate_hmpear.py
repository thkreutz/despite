import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
from types import SimpleNamespace
from src.scheduler import WarmupMultiStepLR
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import time
import torch.nn.functional as F
import wandb
from src.dataset.hmpear import HMPEAR
from src.models import model_loader
from src.models import classifier
from src.models import SPITE

import os

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
    model.train()
    #metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    #metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    total_loss = 0
    num_batches = 0
    
    correct_predictions = 0
    total_samples = 0
    
    header = 'Epoch: [{}]'.format(epoch)
    for clip, target in tqdm(data_loader):
        clip, target = clip.to(device), target.long().to(device)
        output = model(clip)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_batches += 1
        total_loss += loss.item()
        
        # train acc
        prob = F.softmax(input=output, dim=1)
        predictions = torch.argmax(prob, dim=1)

        # Step 3: Update the count of correct predictions
        correct_predictions += (predictions == target).sum().item()
        total_samples += target.size(0)  # Update total sample count
        
        if lr_scheduler:
            lr_scheduler.step()
        
    accuracy = correct_predictions / total_samples
        
    #print("Epoch=%s, loss=%s"% (epoch, total_loss / num_batches))
        #acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        #batch_size = clip.shape[0]
        #metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        #metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        #metric_logger.meters['clips/s'].update(batch_size / (time.time() - start_time))
        #lr_scheduler.step()
        #sys.stdout.flush()
    return accuracy, total_loss / num_batches


def evaluate(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for clip, target in data_loader:
            clip = clip.to(device, non_blocking=True)
            target = target.long().to(device, non_blocking=True)
            output = model(clip)

            prob = F.softmax(input=output, dim=1)
            predictions = torch.argmax(prob, dim=1)
            
            # Step 3: Update the count of correct predictions
            correct_predictions += (predictions == target).sum().item()
            total_samples += target.size(0)  # Update total sample count
            all_preds.extend(predictions.cpu())
            all_labels.extend(target.cpu())
    accuracy = correct_predictions / total_samples
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, conf_matrix

import sys
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HMPEAR CLS Finetuning')
    parser.add_argument('--model_type', default='SPITE', type=str, help='which model to use')
    parser.add_argument('--pretrained_path', default="no", type=str, help='which model to use, no input means train from scratch')
    parser.add_argument('--dataset', default="v1", type=str, help='pretrained on which dataset')

    parser.add_argument('--modality', default="pc", type=str, help='which modality to train on')

    parser.add_argument('--n_frames', default=24, type=int, help='number of frames')
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size')
    parser.add_argument('--epochs', default=35, type=int, help='Epochs')
    parser.add_argument('--embed_dim', default=512, type=int, help='embed dim')

    parser.add_argument('--freeze', default=0, type=int, help='freeze weights and add projection layer')
    parser.add_argument('--projection', default="linear", type=str, help='which projection layer to use')

    parser.add_argument('--wandb', default=0, type=int, help='with wandb logging or no')

    
    input_args = parser.parse_args()

    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Running with configuration:")
    print(f"Model Type      : {input_args.model_type}")
    print(f"Pretrained Path : {input_args.pretrained_path if input_args.pretrained_path else 'Training from scratch'}")
    print(f"Number of Frames: {input_args.n_frames}") # valid = 10, 16, 24
    print(f"Batch Size      : {input_args.batch_size}")
    print(f"Epochs          : {input_args.epochs}")
    print(f"Embed Dim       : {input_args.embed_dim}")
    print(f"Freeze          : {input_args.freeze}")
    print(f"Probing         : {input_args.projection}")
    print(f"Wandb logging   : {input_args.wandb}")

    train = torch.load("/data/LIPD/hmpear_train_1024_%sf.pt" % input_args.n_frames)
    test = torch.load("/data/LIPD/hmpear_test_1024_%sf.pt" % input_args.n_frames)

    X_train = train["PCD"]
    y_train = train["labels"]
    # labels to 
    X_test = test["PCD"]
    y_test = test["labels"]

    label_to_int = {label : i for i,label in enumerate(np.unique(y_train))}
    int_to_label = {i : label for label, i in label_to_int.items()}
    y_train = torch.Tensor([label_to_int[y] for y in y_train])
    y_test = torch.Tensor([label_to_int[y] for y in y_test])


    train_dataset = HMPEAR(X_train, y_train, augment=True)
    test_dataset = HMPEAR(X_test, y_test, augment=False)
    trainLoader = DataLoader(train_dataset, batch_size=input_args.batch_size, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=input_args.batch_size, shuffle=False)

    model_type =  input_args.model_type
    load_pretrained = input_args.pretrained_path 
    embed_dim = input_args.embed_dim
    num_joints = 24
    n_feats = 3
    ##### MODELS - always the same, just init.
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

    n_classes = len(label_to_int)
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
    warmup_iters = lr_warmup_epochs * len(trainLoader)
    lr_milestones = [len(trainLoader) * m for m in lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    with_wandb = input_args.wandb

    if with_wandb:
        # Initialize WandB
        wandb.init(project="SPITE_EVAL_HMPEAR", config={
            "num_points" : 1024,
            "batch_size" : input_args.batch_size,
            "dataset" : input_args.dataset,
            "n_epochs" : input_args.epochs,
            "num_frames" : input_args.n_frames,
            "embed_dim" : input_args.embed_dim,
            "pretrained" : input_args.pretrained_path,
            "model" : input_args.model_type,
            "freeze" : input_args.freeze,
            "projection" : input_args.projection
        })

    print("Start training")
    device = "cuda"
    start_time = time.time()
    
    acc_train_hist = []
    acc_test_hist = []
    loss_hist = []
    acc = 0
    for epoch in range(0, input_args.epochs):
        acc_train, loss = train_one_epoch(classifier_model, criterion, optimizer, lr_scheduler, trainLoader, device, epoch, 5)
        acc_test, _ = evaluate(classifier_model, testLoader, "cuda")

        acc_train_hist.append(acc_train)
        loss_hist.append(loss)
        acc_test_hist.append(acc_test)
        print("Epoch %s | loss= %s - acc_train= %s - acc_test= %s " % (epoch, loss, acc_train, acc_test))
        
        acc = max(acc, acc_test)
        if acc_test == acc: # if max accuracy has changed to this one, save the model so we end up with best cls model.
            if with_wandb:
                torch.save(classifier_model.state_dict(), os.path.join(wandb.run.dir, "cls_model.pth"))

        #acc = max(acc, evaluate(model, criterion, data_loader_test, device=device))
        if with_wandb:
            wandb.log({"epoch": epoch + 1, 
                        "loss": loss,
                        "acc_train" : acc_train,
                        "acc_test" : acc_test,
                        "acc_max" : acc
                            }
                        )

    acc, cm = evaluate(classifier_model, testLoader, "cuda")

    print("Test acc final epoch:", acc)
    print("Max test acc", max(acc_test_hist))
