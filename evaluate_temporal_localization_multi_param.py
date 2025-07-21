from src.models import model_loader
import pickle
import numpy as np
import torch
import argparse
import sys
from src.evaluation import temporal_localization
from src.evaluation import matching
from src.models import SPITE
import os


model_type_to_modalities = {
    "S" : "SKELETON",
    "I" : "IMU",
    "P" : "PC"
}

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Matching Evaluation')
    parser.add_argument('--model_type', default='SPITE', type=str, help='which encoder to use')
    parser.add_argument('--pretrained_path', default="no", type=str, help='which model to use')
    parser.add_argument('--n_frames', default=24, type=int, help='random frames')
    parser.add_argument('--dataset', default="v1", type=str, help='pretrained on which dataset')
    #### experimnt args
    parser.add_argument('--num_windows', default=8, type=int, help='Number of artifical subjects')
    parser.add_argument('--window_size', default=4, type=int, help='Temporal window for matching algorithm')
    parser.add_argument('--n_scenes', default=100, type=int, help='Number of artifical scenes, i.e., matching experiments')
    parser.add_argument('--src_modality', default='imu', type=str, help='Source modality that must be matched to one out of N-targets')
    parser.add_argument('--tgt_modality', default='pc', type=str, help='Target modality that serves as matching candidates')
    parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dim of the model')
    input_args = parser.parse_args()
    
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data
    print("Loading data.")
    with open('../LIPD_SEQUENCES_256p.pkl', 'rb') as f:
        sequence_datasets = pickle.load(f)

     # Map to uppercase because thats how it is encoded. maybe change in future.
    input_args.src_modality = input_args.src_modality.upper()
    input_args.tgt_modality = input_args.tgt_modality.upper()

    #### Load models
    embed_dim = input_args.embed_dim
    num_joints = 24 # keep this the same because we have only one dataset.
    n_feats = 3 # keep this the same because we have only one dataset.

    # map to modalities based on modeltype
    modalities = [model_type_to_modalities[m].lower() for m in input_args.model_type if not m in ["E", "T"]]
    print(modalities)

    ### Load all models, feed into binder model for training later.
    skeleton = model_loader.load_skeleton_encoder(embed_dim, num_joints, n_feats, device="cuda") if "skeleton" in modalities else None
    imu = model_loader.load_imu_encoder(embed_dim, device="cuda") if "imu" in modalities else None
    pc = model_loader.load_pst_transformer(embed_dim, device="cuda") if "pc" in modalities else None
    skeleton_gen = None #model_loader.load_skeleton_generator(embed_dim, num_joints, n_feats, device="cuda") if input_args.with_generator else None

    ### Init the binder model, done.
    model = SPITE.instantiate_binder(modalities, False, imu, pc, skeleton, skeleton_gen).to("cuda")

    if input_args.pretrained_path != "no":
        model.load_state_dict(torch.load(input_args.pretrained_path))
    
    # model.load_state_dict(torch.load("wandb/run-20250217_170137-nffpjmt8/files/model_75.pth"))

    #binder_xa = lidar_bind.BINDER_XA(imu, pc, smpl).to("cuda")
    #binder_xa.load_state_dict(torch.load("wandb/run-20250217_232451-9rfhim7l/files/model_75.pth"))

    print("Encoding data.")
    test_subj_datasets = {}
    for m in ["eLIPD", "eTC", "eDIP"]:
        test_subj_dataset = matching.encode_all(sequence_datasets[m], model, window_length=24, model_type=input_args.model_type)
        test_subj_datasets[m] = test_subj_dataset
    

    print("")
    print("*******************************")
    print("Starting temporal localization experiment MODEL=%s - *** %s -> %s ***" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))
    print("*******************************")
    print("")
    all_diffs = temporal_localization.compute_diffs(test_subj_datasets, src_modality=input_args.src_modality, trgt_modality=input_args.tgt_modality)
    

    if not os.path.exists("results/ICCV/temporal_localization"):
        os.makedirs("results/ICCV/temporal_localization")
    
    if input_args.pretrained_path != "no":
        torch.save(all_diffs, "results/ICCV/temporal_localization/results_diffs_%s_%s_%s.pt" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))
    else:
        torch.save(all_diffs, "results/ICCV/temporal_localization/results_diffs_%s_%s_%s.pt" % ("random", input_args.src_modality, input_args.tgt_modality))
