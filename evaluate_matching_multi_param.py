from src.evaluation import matching
from src.models import model_loader
import pickle
import numpy as np
import torch
import argparse
import sys
import os
from src.models import SPITE

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
    parser.add_argument('--dataset', default="v1", type=str, help='pretrained on which dataset')

    parser.add_argument('--n_frames', default=24, type=int, help='random frames')

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

    print("Encoding data.")
    test_subj_datasets = {}
    for m in ["eLIPD", "eTC", "eDIP"]:
        test_subj_dataset = matching.encode_all(sequence_datasets[m], model, window_length=24, model_type=input_args.model_type)
        test_subj_datasets[m] = test_subj_dataset
    

    ##### Can just run all kinds of combinates here now.
    
    n_subjects_params = [2, 4, 8, 12, 16, 20, 24, 28, 32] #, 48, 64, 128]
    n_window_sizes_params = [1, 2, 4] #[1, 2, 4, 8] #, 12, 16]
    n_scenes_params = [input_args.n_scenes] # repeat same experiment 10k times

    results_dict = {}
    for n_subjects in n_subjects_params:
        for n_window_size in n_window_sizes_params:
            for n_scenes in n_scenes_params:
                print("")
                print("*******************************")
                print("Starting experiment MODEL=%s - *** %s -> %s ***" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))
                print("#Subjects per scene: %s" % n_subjects)
                print("Temporal matching window: %s" % n_window_size)
                print("#Artifical scenes: %s" % n_scenes)
                print("*******************************")
                print("")
                
                run_results = {}
                #### LidarBind
                for m, test_subj_dataset in test_subj_datasets.items(): 
                    ### Get N Random sequences
                    augmented_scenes = matching.create_augmented_scenes_with_windows(test_subj_dataset, num_windows=n_subjects, 
                                                                                    window_size=n_window_size, n_scenes=n_scenes, 
                                                                                    src_modality=input_args.src_modality, tgt_modality=input_args.tgt_modality)
                    results = matching.eval_scenes(augmented_scenes, src_modality=input_args.src_modality, tgt_modality=input_args.tgt_modality)
                    from sklearn.metrics import accuracy_score
                    avg_acc = []
                    for res in results.values():
                        avg_acc.append(accuracy_score(res[0], res[1]))

                    print("%s - Average accuracy over all scenes: %s" % (m, np.mean(avg_acc)))

                    run_results[m] = avg_acc  # dont compute here already so we can get std in plot #np.mean(avg_acc)
                
                results_dict[(n_subjects, n_window_size, n_scenes)] = run_results

    if not os.path.exists("results/ICCV/matching"):
        os.makedirs("results/ICCV/matching")

    torch.save(results_dict, "results/ICCV/matching/results_matching_%s_%s_%s.pt" % (input_args.model_type, input_args.src_modality, input_args.tgt_modality))