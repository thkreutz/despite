import clip
from src.models import SPITE
from src.models import model_loader
from src.models.loss import InfoNCE
import pickle
from tqdm import tqdm
import wandb
import argparse
from types import SimpleNamespace
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.dataset import lipd_babelv1
from src.dataset import lipd_babelv2
import inspect
import itertools


def loss_fn_generator(x, x_hat):
    loss = F.mse_loss(x, x_hat, reduction='mean')
    return loss


def forward_with_kwargs(model, batch):
    """
    Automatically unpacks the batch and passes the correct modalities to the model's forward method.
    
    Args:
    - model: The model instance.
    - batch (dict): The input batch containing the required modalities.
    
    Returns:
    - The output of the model's forward pass.
    """
    
    # Get the argument names of the forward method
    forward_signature = inspect.signature(model.forward)
    forward_args = [param.name for param in forward_signature.parameters.values()]
    
    # Create a kwargs dictionary mapping the batch modalities to the forward arguments
    # Only pass the batch items that match the forward method's arguments
    model_kwargs = {arg: batch[arg].to("cuda") if arg != "with_text" else batch[arg] for arg in forward_args if arg in batch}
    
    # Call the model's forward method using **kwargs
    return model(**model_kwargs)


def train_with_clip(model, clip_model, loss_fun, data_loader, epochs, optimizer, scheduler, modalities, with_generator=False, with_wandb=False):

    # needed for loss computation
    
    modality_combinations = [(m1, m2) for m1, m2 in itertools.combinations(modalities, 2)]
    print(modality_combinations)
    #loss_hist = []
    #acc_hist = []

    for epoch in range(epochs):

        # Init loss tracker
        total_loss_tracker = {
            "total_loss" : 0,
        }
        
        for m1, m2 in modality_combinations:
            total_loss_tracker["%s_%s_loss" % (m1, m2)] = 0
        
        if with_generator:
            for m in modalities:
                if m != "text":
                    total_loss_tracker["%s_gen" % m] = 0
        
        # Init acc tracker
        total_acc_tracker = {"%s_%s_acc" % (m1, m2) : 0 for m1 in modalities for m2 in modalities if m1 != m2}

        #loss_tracker = {}
        num_batches = len(data_loader)
        
        for batch in tqdm(data_loader):

            #### Batch icludes a dictionary
            ## {"imu", "pc", "skeleton", "text", "text_mask"}
            batch["with_text"] = False # generate skeleton from text if we use gen model, we DONT do it now
            text_mask = np.array(batch["text_mask"])
            #print(text_mask)
            #print(sum(text_mask))
            text_tokens = clip.tokenize(batch["batch_text"]).to("cuda")
            emb_text = clip_model.encode_text(text_tokens[text_mask == 1]).float()
            batch["batch_text"] = emb_text
            # get data
            
            # encode each modality except text in binder model, generate from text though
            #out = model(imu, pc, skeleton, emb_text, with_text=True)
            out = forward_with_kwargs(model, batch)
   
            # Loss on the specific modalities...
            # modalities = ["imu", "pc", "text"] 


            # 1. Bind all to CLIP embeddings.

            loss_tracker = {}
            loss_clip = 0
            for m in modalities:
                if m != "text":
                    #  for text, need to apply the mask, leave out text to motion generation for now.
                    loss_temp = loss_fun(query=out[m][text_mask == 1], positive_key=emb_text)
                    loss_clip += loss_temp
                    loss_tracker["%s_text_loss" % m] = loss_temp.item()

            # 2. bind between each other as well
            loss_modalities = 0
            for m1, m2 in modality_combinations:
                if m1 != "text" and m2 != "text": # compute all except for towards text
                    loss_temp = loss_fun(query=out[m1], positive_key=out[m2])
                    loss_modalities += loss_temp
                    loss_tracker["%s_%s_loss" % (m1, m2)] = loss_temp.item()

            # Generator
            # given embeddings of PC -> Generate the Skeleton


            if with_generator:
                loss_generator = 0
                for m in modalities:
                    if m != "text":
                        #print(batch["batch_skeleton"].to("cuda").shape)
                        #print(out["gen_%s" % m].shape)
                        loss_temp = loss_fn_generator(batch["batch_skeleton"].to("cuda"), out["gen_%s" % m])
                        loss_generator += loss_temp
                        loss_tracker["%s_gen" % m] = loss_temp.item()
                #loss_imu_gen = loss_generator(skeleton, out["gen_imu"])
                #loss_pc_gen = loss_generator(skeleton, out["gen_pc"])
                #loss_skeleton_gen = loss_generator(skeleton, out["gen_skeleton"])

            # => generation from text should not be trained because we do not have a distribution, it might skew the training
            #loss_text_gen = loss_generator(skeleton[text_mask == 1], out["gen_text"]) 
            if with_generator:
                loss = 0.4 * loss_clip + 0.4 * loss_modalities + 0.2 * loss_generator
            else:
                loss = 0.5 * loss_clip + 0.5 * loss_modalities #+ ...
                ## loss_Clip => loss_imu_text+loss_pc_text+loss_skeleton_text 
                ## loss_modality => loss_imu_pc+loss_pc_skeleton+loss_skeleton_imu
            loss_tracker["total_loss"] = loss.item()

            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            ### Compute batch acc and log everything...
            acc_tracker = {}
            for m1 in modalities:
                for m2 in modalities:
                    if m1 != m2:
                        if m1 != "text" and m2 != "text":
                            m1_acc, m2_acc = evaluate_batch_similarity(out[m1], out[m2], device="cuda")
                        else:
                            if m1 == "text":
                                m1_acc, m2_acc = evaluate_batch_similarity(emb_text, out[m2][text_mask == 1], device="cuda")
                            else:
                                m1_acc, m2_acc = evaluate_batch_similarity(out[m1][text_mask == 1], emb_text, device="cuda")
                        acc_tracker["%s_%s_acc" % (m1, m2)] = m1_acc.item()
                        acc_tracker["%s_%s_acc" % (m2, m1)] = m2_acc.item()

            #print(loss_tracker)
            #print(total_loss_tracker)
            #print(acc_tracker)
            #print(total_acc_tracker)
            # Update loss and acc tracker
            total_loss_tracker = {k : loss_tracker[k] + total_loss_tracker[k] for k, v in total_loss_tracker.items()}
            total_acc_tracker = {k : acc_tracker[k] + total_acc_tracker[k] for k, v in total_acc_tracker.items()}
            #break
        print(total_acc_tracker)
        print(num_batches)
        log = {**{k : v / num_batches for k, v in total_loss_tracker.items()}, **{k : v / num_batches for k, v in total_acc_tracker.items()}}
        log["epoch"] = epoch + 1
        log["lr"] = optimizer.param_groups[0]['lr']
        if with_wandb:
            wandb.log(log)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_%s.pth" % epoch))
        else:
            print(log)
            #print(f"Average Loss for Epoch {epoch+1}: {total_loss / num_batches:.4f}")
            
        #loss_hist.append(total_loss / num_batches)
        #acc_hist.append([np.mean(s_t_accuracies), np.mean(t_s_accuracies)])
        #print(f"Average Loss for Epoch {epoch+1}: {total_loss / num_batches:.4f}")
    #return loss_hist, acc_hist

def train_without_clip(model, loss_fun, data_loader, epochs, optimizer, scheduler, modalities, with_generator=False, with_wandb=False):

    # needed for loss computation
    
    modality_combinations = [(m1,m2) for m1, m2 in itertools.combinations(modalities, 2)]
    print(modality_combinations)
    #loss_hist = []
    #acc_hist = []

    for epoch in range(epochs):

        # Init loss tracker
        total_loss_tracker = {
            "total_loss" : 0,
        }
        
        for m1, m2 in modality_combinations:
            total_loss_tracker["%s_%s_loss" % (m1, m2)] = 0
        
        if with_generator:
            for m in modalities:
                if m != "text":
                    total_loss_tracker["%s_gen" % m] = 0
        
        # Init acc tracker
        total_acc_tracker = {"%s_%s_acc" % (m1, m2) : 0 for m1 in modalities for m2 in modalities if m1 != m2}

        #loss_tracker = {}
        num_batches = len(data_loader)
        
        for batch in tqdm(data_loader):

            #### Batch icludes a dictionary
            ## {"imu", "pc", "skeleton", "text", "text_mask"}
            batch["with_text"] = False # generate skeleton from text if we use gen model, we DONT do it now
            #text_mask = np.array(batch["text_mask"])
            #print(text_mask)
            #print(sum(text_mask))
            #text_tokens = clip.tokenize(batch["batch_text"]).to("cuda")
            #emb_text = clip_model.encode_text(text_tokens[text_mask == 1]).float()
            batch["batch_text"] = None
            # get data
            
            # encode each modality except text in binder model, generate from text though
            #out = model(imu, pc, skeleton, emb_text, with_text=True)
            out = forward_with_kwargs(model, batch)
   
            # Loss on the specific modalities...
            # modalities = ["imu", "pc", "text"] 


            # 1. Bind all to CLIP embeddings.

            loss_tracker = {}

            #loss_clip = 0
            #for m in modalities:
            #    if m != "text":
            #        #  for text, need to apply the mask, leave out text to motion generation for now.
            #        loss_temp = loss_fun(query=out[m][text_mask == 1], positive_key=emb_text)
            #        loss_clip += loss_temp
            #        loss_tracker["%s_text_loss" % m] = loss_temp.item()

            # 2. bind between each other as well
            loss_modalities = 0
            for m1, m2 in modality_combinations:
                if m1 != "text" and m2 != "text": # compute all except for towards text
                    loss_temp = loss_fun(query=out[m1], positive_key=out[m2])
                    loss_modalities += loss_temp
                    loss_tracker["%s_%s_loss" % (m1, m2)] = loss_temp.item()

            # Generator
            # given embeddings of PC -> Generate the Skeleton

            if with_generator:
                loss_generator = 0
                for m in modalities:
                    if m != "text":
                        #print(batch["batch_skeleton"].to("cuda").shape)
                        #print(out["gen_%s" % m].shape)
                        loss_temp = loss_fn_generator(batch["batch_skeleton"].to("cuda"), out["gen_%s" % m])
                        loss_generator += loss_temp
                        loss_tracker["%s_gen" % m] = loss_temp.item()
                #loss_imu_gen = loss_generator(skeleton, out["gen_imu"])
                #loss_pc_gen = loss_generator(skeleton, out["gen_pc"])
                #loss_skeleton_gen = loss_generator(skeleton, out["gen_skeleton"])

            # => generation from text should not be trained because we do not have a distribution, it might skew the training
            #loss_text_gen = loss_generator(skeleton[text_mask == 1], out["gen_text"]) 
            if with_generator:
                loss = 0.8 * loss_modalities + 0.2 * loss_generator
            else:
                loss = loss_modalities

            loss_tracker["total_loss"] = loss.item()

            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            ### Compute batch acc and log everything...
            acc_tracker = {}
            for m1 in modalities:
                for m2 in modalities:
                    if m1 != m2:
                        m1_acc, m2_acc = evaluate_batch_similarity(out[m1], out[m2], device="cuda")
                        acc_tracker["%s_%s_acc" % (m1, m2)] = m1_acc.item()
                        acc_tracker["%s_%s_acc" % (m2, m1)] = m2_acc.item()

            #print(loss_tracker)
            #print(total_loss_tracker)
            #print(acc_tracker)
            #print(total_acc_tracker)
            # Update loss and acc tracker
            total_loss_tracker = {k : loss_tracker[k] + total_loss_tracker[k] for k, v in total_loss_tracker.items()}
            total_acc_tracker = {k : acc_tracker[k] + total_acc_tracker[k] for k, v in total_acc_tracker.items()}
            #break
        print(total_acc_tracker)
        print(num_batches)
        log = {**{k : v / num_batches for k,v in total_loss_tracker.items()}, **{k : v / num_batches for k,v in total_acc_tracker.items()}}
        log["epoch"] = epoch + 1
        log["lr"] = optimizer.param_groups[0]['lr']
        if with_wandb:
            wandb.log(log)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_%s.pth" % epoch))
        else:
            print(log)
            #print(f"Average Loss for Epoch {epoch+1}: {total_loss / num_batches:.4f}")
            
        #loss_hist.append(total_loss / num_batches)
        #acc_hist.append([np.mean(s_t_accuracies), np.mean(t_s_accuracies)])
        #print(f"Average Loss for Epoch {epoch+1}: {total_loss / num_batches:.4f}")
    #return loss_hist, acc_hist

def evaluate_batch_similarity(source_embeddings, target_embeddings, device):
    """
    Given a batch matrix (size B) of paired embeddings,
    measure the accuracy of the predictions by checking nearest the neighbors
    """
    #  Compute similarity
    s = torch.nn.functional.normalize(source_embeddings, dim=1)
    t = torch.nn.functional.normalize(target_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy, t_s_accuracy

import sys

if __name__ == "__main__":
    

    ###### ARGS
    parser = argparse.ArgumentParser(description='SPITE TRAINING')
    #parser.add_argument('--pointcloud_encoder', default='psttransformer', type=str, help='which pointcloud encoder to use')
    parser.add_argument('--pretrained_path', default="no", type=str, help='Loading weights? "no" input means train from scratch')

    parser.add_argument('--n_frames', default=24, type=int, help='number of frames')
    parser.add_argument('--batch_size', default=2028, type=int, help='Batch size')
    parser.add_argument('--epochs', default=150, type=int, help='Epochs')
    parser.add_argument('--embed_dim', default=512, type=int, help='embedding dimension')
    #parser.add_argument('--modalities', default=["pc", "imu", "skeleton", "text"], type=str, help='which modalities to train on')
    parser.add_argument('--modalities', nargs='+', help='List of modalities', required=True)
    parser.add_argument('--with_generator', default=0, type=int, help='with skeleton generator')
    parser.add_argument('--wandb', default=0, type=int, help='with wandb logging or no')

    parser.add_argument('--dataset', default="v1", type=str, help='Which dataset to use')
    input_args = parser.parse_args()

    ### Embed dim when using text MUST be 512
    # =>
    if "text" in input_args.modalities:
        assert input_args.embed_dim == 512, "embedding dim must be 512 when using text"
        

    print(f"Running with configuration:")
    print(f"Dataset         : {input_args.dataset}")
    print(f"Model Type      : {SPITE.get_name(input_args.modalities, input_args.with_generator)}")
    print(f"Modalities      : {input_args.modalities}")
    print(f"Pretrained Path : {input_args.pretrained_path if input_args.pretrained_path else 'Training from scratch'}")
    print(f"Number of Frames: {input_args.n_frames}")
    print(f"Batch Size      : {input_args.batch_size}")
    print(f"Epochs          : {input_args.epochs}")
    print(f"Embed Dim       : {input_args.embed_dim}")

    modalities = input_args.modalities

    if "text" in modalities:
        # Text encoder is always CLIP.
        print("Loading models...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda", jit=False)  # Must set jit=False for training

        for p in clip_model.parameters():
            p.requires_grad = False

    embed_dim = input_args.embed_dim
    num_joints = 24 # keep this the same because we have only one dataset.
    n_feats = 3 # keep this the same because we have only one dataset.
    
    ### Load all models, feed into binder model for training later.
    skeleton = model_loader.load_skeleton_encoder(embed_dim, num_joints, n_feats, device="cuda") if "skeleton" in modalities else None
    imu = model_loader.load_imu_encoder(embed_dim, device="cuda") if "imu" in modalities else None
    pc = model_loader.load_pst_transformer(embed_dim, device="cuda") if "pc" in modalities else None
    skeleton_gen = model_loader.load_skeleton_generator(embed_dim, num_joints, n_feats, device="cuda") if input_args.with_generator else None

    ### Init the binder model, done.
    binder = SPITE.instantiate_binder(modalities, input_args.with_generator, imu, pc, skeleton, skeleton_gen).to("cuda")
    
    print("Loading data...")
    with open("/data/LIPD/lipd_babel_annotations.pkl", "rb") as f:
        seqs_with_labels_train = pickle.load(f)

    with open("/data/LIPD/lipd_babel_annotations_VAL.pkl", "rb") as f:
        seqs_with_labels_val = pickle.load(f)

    with open('/data/LIPD/LIPD_SEQUENCES_256p.pkl', 'rb') as f:
        sequence_datasets = pickle.load(f)

    #### Dataset
    if input_args.dataset == "v1":
        print("Loading LIPD-Babel-v1")
        dataset = lipd_babelv1.LIPDBabelv1(sequence_datasets, 
                                            seqs_with_labels_train, 
                                            seqs_with_labels_val, 
                                            num_frames=input_args.n_frames, 
                                            augment=True, 
                                            train=True, 
                                            modalities=modalities)

    if input_args.dataset == "v2":
        print("Loading LIPD-Babel-v2")
        dataset = lipd_babelv2.LIPDBabelv2(sequence_datasets, 
                                            seqs_with_labels_train, 
                                            seqs_with_labels_val, 
                                            num_frames=input_args.n_frames, 
                                            augment=True, 
                                            train=True, 
                                            modalities=modalities)
    
    batch_size = input_args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set learning rate and optimizer
    optimizer = torch.optim.Adam(binder.parameters(), lr=1e-4) #, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # Could set a learning rate scheduler
    total_steps = len(data_loader) * input_args.epochs
    scheduler = None
    #scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=int(0.1 * total_steps),
    #                                            num_training_steps=total_steps)
    
    loss_fun = InfoNCE(symmetric_loss=True, learn_temperature=False)



    with_wandb = input_args.wandb
    if with_wandb:
        # Initialize WandB
            wandb.init(project="SPITE", config={
                "num_points" : 256, # fixed subsampling.
                "batch_size" : input_args.batch_size,
                "embed_dim" : input_args.embed_dim,
                "n_epochs" : input_args.epochs,
                "num_frames" : input_args.n_frames,
                "dataset" : input_args.dataset,
                "Model Type" : SPITE.get_name(input_args.modalities, input_args.with_generator),
                "Modalities" : input_args.modalities,
                "Pretrained Path" : input_args.pretrained_path if input_args.pretrained_path else 'Training from scratch',
                "with_generator" : input_args.with_generator
                                        })
    if "text" in modalities:
        train_with_clip(binder, clip_model, loss_fun, data_loader, input_args.epochs, optimizer, scheduler, modalities, input_args.with_generator, with_wandb)
    else:
        train_without_clip(binder, loss_fun, data_loader, input_args.epochs, optimizer, scheduler, modalities, input_args.with_generator, with_wandb)
