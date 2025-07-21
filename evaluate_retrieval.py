from src.models import model_loader
from src.models import lidar_bind
import pickle
import numpy as np
import torch
import argparse
import sys
from src.models import tmr_text_encoder
from src.models import text
import clip
from src.dataset import sequence_dataset
import importlib
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.evaluation import retrieval
importlib.reload(sequence_dataset)

def action_labels():
    action_label_to_idx = {
        "walk": 0,
        "stand": 1,
        "hand movements": 2,
        "turn": 3,
        "interact with/use object": 4,
        "arm movements": 5,
        "t pose": 6,
        "step": 7,
        "backwards movement": 8,
        "raising body part": 9,
        "look": 10,
        "touch object": 11,
        "leg movements": 12,
        "forward movement": 13,
        "circular movement": 14,
        "stretch": 15,
        "jump": 16,
        "touching body part": 17,
        "sit": 18,
        "place something": 19,
        "take/pick something up": 20,
        "run": 21,
        "bend": 22,
        "throw": 23,
        "foot movements": 24,
        "a pose": 25,
        "stand up": 26,
        "lowering body part": 27,
        "sideways movement": 28,
        "move up/down incline": 29,
        "action with ball": 30,
        "kick": 31,
        "gesture": 32,
        "head movements": 33,
        "jog": 34,
        "grasp object": 35,
        "waist movements": 36,
        "lift something": 37,
        "knee movement": 38,
        "wave": 39,
        "move something": 40,
        "swing body part": 41,
        "catch": 42,
        "dance": 43,
        "lean": 44,
        "greet": 45,
        "poses": 46,
        "touching face": 47,
        "sports move": 48,
        "exercise/training": 49,
        "clean something": 50,
        "punch": 51,
        "squat": 52,
        "scratch": 53,
        "hop": 54,
        "play sport": 55,
        "stumble": 56,
        "crossing limbs": 57,
        "perform": 58,
        "martial art": 59,
        "balance": 60,
        "kneel": 61,
        "shake": 62,
        "grab body part": 63,
        "clap": 64,
        "crouch": 65,
        "spin": 66,
        "upper body movements": 67,
        "knock": 68,
        "adjust": 69,
        "crawl": 70,
        "twist": 71,
        "move back to original position": 72,
        "bow": 73,
        "hit": 74,
        "touch ground": 75,
        "shoulder movements": 76,
        "telephone call": 77,
        "grab person": 78,
        "play instrument": 79,
        "tap": 80,
        "spread": 81,
        "skip": 82,
        "rolling movement": 83,
        "jump rope": 84,
        "play catch": 85,
        "drink": 86,
        "evade": 87,
        "support": 88,
        "point": 89,
        "side to side movement": 90,
        "stop": 91,
        "protect": 92,
        "wrist movements": 93,
        "stances": 94,
        "wait": 95,
        "shuffle": 96,
        "lunge": 97,
        "communicate (vocalise)": 98,
        "jumping jacks": 99,
        "rub": 100,
        "dribble": 101,
        "swim": 102,
        "sneak": 103,
        "to lower a body part": 104,
        "misc. abstract action": 105,
        "mix": 106,
        "limp": 107,
        "sway": 108,
        "slide": 109,
        "cartwheel": 110,
        "press something": 111,
        "shrug": 112,
        "open something": 113,
        "leap": 114,
        "trip": 115,
        "golf": 116,
        "move misc. body part": 117,
        "get injured": 118,
        "sudden movement": 119,
        "duck": 120,
        "flap": 121,
        "salute": 122,
        "stagger": 123,
        "draw": 124,
        "tie": 125,
        "eat": 126,
        "style hair": 127,
        "relax": 128,
        "pray": 129,
        "flip": 130,
        "shivering": 131,
        "interact with rope": 132,
        "march": 133,
        "zombie": 134,
        "check": 135,
        "wiggle": 136,
        "bump": 137,
        "give something": 138,
        "yoga": 139,
        "mime": 140,
        "wobble": 141,
        "release": 142,
        "wash": 143,
        "stroke": 144,
        "rocking movement": 145,
        "swipe": 146,
        "strafe": 147,
        "hang": 148,
        "flail arms": 149
    }
    idx_to_action_label = {}
    for key, value in action_label_to_idx.items():
        idx_to_action_label[value] = key
        
    action_text_labels = list(action_label_to_idx.keys())
    action_text_labels.sort(key=lambda x: action_label_to_idx[x])

    return action_label_to_idx, idx_to_action_label, action_text_labels

def motion_to_text(dataset, bdg, action_label_to_idx, classes_text_emb_norm):
    ##### Can just run all kinds of combinates here now.
    iterator = DataLoader(dataset, batch_size=128, shuffle=False) 

    correct_preds_top_5 = { "smpl" : 0, "pc" : 0, "imu" : 0}
    correct_preds_top_1 = { "smpl" : 0, "pc" : 0, "imu" : 0}
    #correct_preds_top_5, 
    #correct_preds_top_1 = 0,0
    total_samples = 0
    print("num batches=",len(iterator))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            pc = batch[0].to("cuda")
            imu = batch[1].to("cuda")
            smpl = batch[2].to("cuda")
            text_labels = batch[3]
            batch = bdg(imu, pc, smpl, None, with_text=False)
            
            # get all categories of the action
            labels = list(map(lambda x: [action_label_to_idx[cat] for cat in x.split(",") if cat in action_label_to_idx], text_labels))
            
            
            #### Compute for each modality
            for act_modality in ["smpl", "pc", "imu"]:
            
                motion_features_norm = batch[act_modality] / batch[act_modality].norm(dim=-1, keepdim=True)
                scores = motion_features_norm @ classes_text_emb_norm.t()
                similarity = (100.0 * motion_features_norm @ classes_text_emb_norm.t()).softmax(dim=-1)

                #total_samples += similarity.shape[0]
                for i in range(similarity.shape[0]):
                    values, indices = similarity[i].topk(5)

                    # TOP-5 CHECK
                    if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                        correct_preds_top_5[act_modality] += 1

                    # TOP-1 CHECK
                    values = values[:1]
                    indices = indices[:1]
                    if any([gt_cat_idx in indices for gt_cat_idx in labels[i]]):
                        correct_preds_top_1[act_modality] += 1

    total_samples = len(dataset)
    # print(f"Current Top-5 Acc. : {100 * correct_preds_top_5 / total_samples:.2f}%")
    print("****** REPORT ******")
    for act_modality in ["smpl", "pc", "imu"]:
        print(f"{act_modality} Top-1 Acc. : {100 * correct_preds_top_1[act_modality] / total_samples:.2f}%  ({correct_preds_top_1[act_modality]}/{total_samples})")
        print(f"{act_modality} Top-5 Acc. : {100 * correct_preds_top_5[act_modality] / total_samples:.2f}%  ({correct_preds_top_5[act_modality]}/{total_samples})")
        correct_preds_top_1[act_modality] = 100 * correct_preds_top_1[act_modality] / total_samples
        correct_preds_top_5[act_modality] = 100 * correct_preds_top_5[act_modality] / total_samples

    return correct_preds_top_1, correct_preds_top_5


def text_to_motion(dataset, bdg, text_model):

    iterator = DataLoader(dataset, batch_size=128, shuffle=False) 

    test_encodings = {}
    test_encodings["smpl"] = []
    test_encodings["imu"] = []
    test_encodings["pc"] = []
    test_encodings["text"] = []
    test_encodings["queries"] = []
    print("num batches=",len(iterator))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            pc = batch[0].to("cuda")
            imu = batch[1].to("cuda")
            smpl = batch[2].to("cuda")
            text_labels = [l.split(",")[0] for l in batch[3]]

            if encode_text_with == "CLIP":
                texts = clip.tokenize(text_labels).to("cuda")
                batch = bdg(imu, pc, smpl, None, with_text=False)
                batch['clip_text_embed'] = text_model.encode_text(texts).float()
            else:
                batch = bdg(imu, pc, smpl, None, with_text=False)
                batch["clip_text_embed"] = retrieval.encode_text_list_with_tmr(text_model[0], text_model[1], text_labels)

            test_encodings["smpl"].append(batch["smpl"] / batch['smpl'].norm(dim=-1, keepdim=True))
            test_encodings["imu"].append(batch["imu"] / batch['imu'].norm(dim=-1, keepdim=True))
            test_encodings["pc"].append(batch["pc"] / batch['pc'].norm(dim=-1, keepdim=True))
            test_encodings["text"].append(batch['clip_text_embed'] / batch['clip_text_embed'].norm(dim=-1, keepdim=True))
            test_encodings["queries"].extend(text_labels)

    text_labels = np.array(test_encodings["queries"])
    transition_filter = torch.Tensor(text_labels != "transition").bool()

    # Stack tensors for batched computation
    test_retrieval = {}
    test_retrieval["smpl"] = torch.cat(test_encodings["smpl"])[transition_filter]  # (N, D)
    test_retrieval["imu"] = torch.cat(test_encodings["imu"])[transition_filter]    # (N, D)
    test_retrieval["pc"] = torch.cat(test_encodings["pc"])[transition_filter]      # (N, D)
    test_retrieval["text"] = torch.cat(test_encodings["text"])[transition_filter]  # (N, D)
    test_retrieval["queries"] = text_labels[transition_filter.numpy()]  # (N, D)

    # Convert text labels into embeddings

    results = {}
    for tgt_mod in ["smpl", "imu", "pc"]:

        retrieved_indices = []
        retrieved_texts = []
        for i in range(0, len(test_retrieval["smpl"]), 10000):
            motion_embeddings = test_retrieval[tgt_mod][i:i+10000]  # (N_queries, D)
            query_embeddings = test_retrieval["text"][i:i+10000]  # (N_queries, D)
            all_text_embeddings = test_retrieval["text"][i:i+10000] # (N_motions, D)

            # Compute cosine similarity (N_queries x N_motions)
            similarity_matrix = torch.mm(query_embeddings, motion_embeddings.T)

            retrieved_indices_temp = similarity_matrix.argmax(dim=1)  # (N_queries,) - highest similarity motion index
            retrieved_texts_temp = torch.vstack([all_text_embeddings[i] for i in retrieved_indices_temp])  #
            
            retrieved_indices.append(retrieved_indices_temp)
            retrieved_texts.append(retrieved_texts_temp)

        retrieved_indices = torch.cat(retrieved_indices)
        retrieved_texts = torch.vstack(retrieved_texts)

        for thresh in [0.8, 0.85, 0.9, 0.95]:
            # Evaluate retrieval correctness
            correct_retrievals = 0
            sim_scores = []
            for query_text, retrieved_text in zip(test_retrieval["text"], retrieved_texts):
                sim_score = retrieval.compute_text_similarity(query_text, retrieved_text)
                sim_scores.append(sim_score)
                if sim_score >= thresh:
                    correct_retrievals += 1

            retrieval_accuracy = correct_retrievals / len(test_retrieval["text"])
            print(f"Retrieval Accuracy Text->{tgt_mod} (Threshold = {thresh}): {retrieval_accuracy:.4f}")

            results[(tgt_mod, thresh)] = retrieval_accuracy
        # Apply threshold (keeping values above 0.95)
        #threshold = 0.95
        #valid_matches = similarity_matrix >= threshold  # Boolean mask
    return results


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Matching Evaluation')
    parser.add_argument('--model_type', default='clia', type=str, help='which encoder to use')
    parser.add_argument('--text_encoder', default='CLIP', type=str, help='Which text encoder to use')
    parser.add_argument('--pretrained_path', default=None, type=str, help='which model to use')
    parser.add_argument('--n_frames', default=24, type=int, help='random frames')
    parser.add_argument('--babel', default=60, type=int, help='Babel 60 or 120')
    #### experimnt args
    parser.add_argument('--src_modality', default='IMU', type=str, help='Source modality that must be matched to one out of N-targets')
    parser.add_argument('--tgt_modality', default='PCD', type=str, help='Target modality that serves as matching candidates')
    parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dimension of the encoders')

    args = parser.parse_args()
    
    # data
    print("Loading data.")
    with open('/data/LIPD/LIPD_SEQUENCES_256p.pkl', 'rb') as f:
        sequence_datasets = pickle.load(f)
    
    with open("/data/LIPD/lipd_babel_annotations_VAL.pkl", "rb") as f:
        seqs_with_labels_val = pickle.load(f)

    dataset = sequence_dataset.SequenceDatasetSlidingWindowWithText_BabelSplit(sequence_datasets, None, seqs_with_labels_val, num_frames=args.n_frames, augment=False, train=False)
    
    
    print("Loading model.")

    # models, params always the same..
    embed_dim = args.embed_dim
    num_joints = 24
    n_feats = 3

    smpl = model_loader.load_smpl_encoder(embed_dim, num_joints, n_feats, device="cuda")
    imu = model_loader.load_imu_encoder(embed_dim, device="cuda")
    pc = model_loader.load_pst_transformer(embed_dim, device="cuda")
    smpl_gen = model_loader.load_smpl_generator(embed_dim, num_joints, n_feats, device="cuda")

    encode_text_with = args.text_encoder
    if encode_text_with == "CLIP":
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda", jit=False)  # Must set jit=False for training

        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        text_emb_model = text.TextToEmb(modelpath="distilbert-base-uncased", device="cuda")
        text_encoder = tmr_text_encoder.ACTORStyleEncoder(nfeats=768, num_layers=6, vae=True, dropout=0).to("cuda")
        text_encoder.load_state_dict(torch.load("../TMR/models/models/tmr_babel_guoh3dfeats/last_weights/text_encoder.pt"))
        #text_encoder.eval()

    ### So far only use BINDERGen
    bdg = lidar_bind.BINDERGen(imu, pc, smpl, smpl_gen).cuda()
    bdg.load_state_dict(torch.load(args.pretrained_path))

    print("Setting in eval mode.")
    bdg.eval()
    
    #if not args.model_type in ["clia", "cliaxa", "clisa", "clisaxa"]:
    #    print("model does not exist. exiting.")
    #    sys.exit()

    #if args.model_type == "clia": 
    #    model = clia.CLIP_IMU_LiDAR(imu, pc).to("cuda")
    #elif args.model_type == "cliaxa":
    #    model = clia.CLIP_IMU_LiDAR_CA(imu, pc).to("cuda")
    #elif args.model_type == "clisa":
    #    model = lidar_bind.BINDER(imu, pc, smpl).to("cuda")
    #else:
    #    model = lidar_bind.BINDER_XA(imu, pc, smpl).to("cuda")

    #model.load_state_dict(torch.load(args.pretrained_path))
    # model.load_state_dict(torch.load("wandb/run-20250217_170137-nffpjmt8/files/model_75.pth"))

    #binder_xa = lidar_bind.BINDER_XA(imu, pc, smpl).to("cuda")
    #binder_xa.load_state_dict(torch.load("wandb/run-20250217_232451-9rfhim7l/files/model_75.pth"))

    action_label_to_idx, idx_to_action_label, action_text_labels = action_labels()

    if encode_text_with == "CLIP":
        texts = clip.tokenize(action_text_labels[:args.babel]).to("cuda")
        classes_text_emb = clip_model.encode_text(texts).float() #.cpu()
        classes_text_emb_norm = classes_text_emb / classes_text_emb.norm(dim=-1, keepdim=True)
    else:
        classes_text_emb = retrieval.encode_text_list_with_tmr(text_emb_model, text_encoder, action_text_labels[:args.babel])
        classes_text_emb_norm = classes_text_emb / classes_text_emb.norm(dim=-1, keepdim=True)

    correct_preds_top_1, correct_preds_top_5 = motion_to_text(dataset, bdg, action_label_to_idx, classes_text_emb_norm)

    if encode_text_with == "CLIP":
        text_model = clip_model
    else:
        text_model = [text_emb_model, text_encoder]
    
    results_text_motion_retrieval = text_to_motion(dataset, bdg, text_model)