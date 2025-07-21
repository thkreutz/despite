import numpy as np
import torch
from tqdm import tqdm

def compute_diffs(test_subj_datasets, src_modality="PCD", trgt_modality="IMU"):
    all_diffs = {}
    testsets = ["eLIPD", "eTC", "eDIP"]
    for m in testsets:
        for subject in test_subj_datasets[m].keys():
            for sid in test_subj_datasets[m][subject].keys():
                sid_lidar_embs = test_subj_datasets[m][subject][sid]["%s_EMBS" % trgt_modality]
                sid_imu_embs = test_subj_datasets[m][subject][sid]["%s_EMBS" % src_modality]

                diffs = []
                for idx in tqdm(range(len(sid_imu_embs))):
                    imu_query_emb = sid_imu_embs[idx]

                    # sims in one shot
                    frame_sims = torch.matmul(sid_lidar_embs, imu_query_emb)  
                    #frame_sims = [1 - cosine(imu_query_emb, lidar_emb) for lidar_emb in sid_lidar_embs]

                    diff = torch.topk(frame_sims, 50).indices - idx
                    #diff = np.abs(np.argmax(frame_sims) - idx) ### TODO: [X] return k-argmax

                    diffs.append(diff)
                all_diffs[(m, subject, sid)] = diffs
    
    return all_diffs
