import torch
import numpy as np
import random
from scipy.spatial.distance import cosine
from tqdm import tqdm
import sys

def preprocess_sliding_windows(point_clouds, imu, smpl, window_length, stride=1):
    n_frames, n_points, dims = point_clouds.shape
    
    # Calculate the number of windows we can extract
    num_windows = n_frames - window_length + 1
    #num_windows = (n_frames - window_length) // stride + 1 if (n_frames - window_length) >= 0 else 0

    # Initialize a tensor to hold the sliding windows
    windows_pcd = []
    windows_imu = []

    windows_smpl = []

    # Populate each window
    for i in range(num_windows):
        start_idx = i * stride
        windows_pcd.append(point_clouds[start_idx:start_idx + window_length])
        windows_imu.append(imu[start_idx:start_idx + window_length])
        windows_smpl.append(smpl[start_idx:start_idx + window_length])

    #print(len(windows))
    #print(windows[0])
    return torch.stack(windows_pcd), torch.stack(windows_imu), torch.stack(windows_smpl)

def encode_all(test_subj_dataset, model, window_length=24, model_type="SPITE", normalize=True):
    # Input is a dictionary with dataset->subject->sequence->{values}

    # #1 transform each sequence into sliding windows of size w
    for subject in test_subj_dataset.keys():
        for seq in test_subj_dataset[subject].keys():

            test_subj_dataset[subject][seq]["PC"] = test_subj_dataset[subject][seq]["PCD"]
            del test_subj_dataset[subject][seq]["PCD"]    
            #print(test_subj_dataset[subject][seq]["gt_joint"].reshape(-1, 24, 3).shape)
            windows_pcs, windows_imus, windows_smpls = preprocess_sliding_windows(test_subj_dataset[subject][seq]["PC"], test_subj_dataset[subject][seq]["IMU"], test_subj_dataset[subject][seq]["gt_joint"].reshape(-1, 24, 3), window_length=window_length)
            test_subj_dataset[subject][seq]["PC_W"] = windows_pcs
            test_subj_dataset[subject][seq]["IMU_W"] = windows_imus
            test_subj_dataset[subject][seq]["SKELETON_W"] = windows_smpls
            test_subj_dataset[subject][seq]["SKELETON"] = test_subj_dataset[subject][seq]["gt_joint"].reshape(-1, 24, 3) #qq

    # #2 Encode each slinding window of each sequence of each subject
    for subject in test_subj_dataset.keys():
        for seq in test_subj_dataset[subject].keys():
            windows_pcs = test_subj_dataset[subject][seq]["PC_W"]
            windows_imus = test_subj_dataset[subject][seq]["IMU_W"]
            windows_smpls = test_subj_dataset[subject][seq]["SKELETON_W"]
            with torch.no_grad():

                ### Models
                # SPI
                # SP
                # SI
                # PI

                if "SPI" in model_type: 
                    out = model(windows_imus.to("cuda"), windows_pcs.to("cuda"), windows_smpls.to("cuda"))
                    ### normalize already here
                    imu_embs = out["imu"].cpu() 
                    pcd_embs = out["pc"].cpu()
                    smpl_embs = out["skeleton"].cpu()

                    if normalize:
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs / np.linalg.norm(imu_embs, axis=1, keepdims=True)
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pcd_embs / np.linalg.norm(pcd_embs, axis=1, keepdims=True)
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = smpl_embs / np.linalg.norm(smpl_embs, axis=1, keepdims=True)
                    else:
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs 
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pcd_embs
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = smpl_embs

                elif "SP" in model_type:
                    # clia no SKELETON encoder
                    out = model(windows_pcs.to("cuda"), windows_smpls.to("cuda"))
                    ### normalize already here
                    skeleton_embs = out["skeleton"].cpu()
                    pcd_embs = out["pc"].cpu()

                    if normalize:
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = skeleton_embs / np.linalg.norm(skeleton_embs, axis=1, keepdims=True)
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pcd_embs / np.linalg.norm(pcd_embs, axis=1, keepdims=True)
                    else:
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = skeleton_embs 
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pcd_embs

                elif "SI" in model_type:
                    out = model(windows_imus.to("cuda"), windows_smpls.to("cuda"))
                    ### normalize already here
                    imu_embs = out["imu"].cpu()
                    skeleton_embs = out["skeleton"].cpu()
                    

                    if normalize:
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = skeleton_embs / np.linalg.norm(skeleton_embs, axis=1, keepdims=True)
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs / np.linalg.norm(imu_embs, axis=1, keepdims=True)
                    else:
                        test_subj_dataset[subject][seq]["SKELETON_EMBS"] = skeleton_embs 
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs

                elif "PI" in model_type:            
                    out = model(windows_imus.to("cuda"), windows_pcs.to("cuda"))
                    ### normalize already here
                    imu_embs = out["imu"].cpu()
                    pc_embs = out["pc"].cpu()
                    

                    if normalize:
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs / np.linalg.norm(imu_embs, axis=1, keepdims=True)
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pc_embs / np.linalg.norm(pc_embs, axis=1, keepdims=True)
                    else:
                        test_subj_dataset[subject][seq]["IMU_EMBS"] = imu_embs 
                        test_subj_dataset[subject][seq]["PC_EMBS"] = pc_embs
                else:
                    print("ERROR, DONT KNOW THIS MODEL")
                    sys.exit(0)

    return test_subj_dataset

def create_augmented_scenes_with_windows(dataset, num_windows=3, window_size=30, n_scenes=10, src_modality="IMU", tgt_modality="PC"):
    """
    Create augmented scenes with sliding windows for multiple-frame matching.
    """
    augmented_scenes = []
    subjects = list(dataset.keys())
    
    for j in range(n_scenes):
        # Select distractors (one window from each distractor subject)
        match_windows_imu = []
        match_windows_pcd = []
        
        match_gt_imus = []
        match_gt_pcds = []
        # num_windows == num other subjects
        
        while len(match_windows_imu) < num_windows:
            random_subject = random.choice(subjects)
            #if random_subject != subject:
            random_sequence = random.choice(list(dataset[random_subject].keys()))
            random_lidar_embs = dataset[random_subject][random_sequence]["%s_EMBS" % tgt_modality]
            random_imu_embs = dataset[random_subject][random_sequence]["%s_EMBS" % src_modality]

            random_gt_lidar_embs = dataset[random_subject][random_sequence]["%s_W"%tgt_modality]
            random_gt_imu_embs = dataset[random_subject][random_sequence]["%s_W"%src_modality]

            # window_size == matching length
            if len(random_lidar_embs) >= window_size:
                start_idx_matchwindow = random.randint(0, len(random_lidar_embs) - window_size)
                match_window_imu = random_imu_embs[start_idx_matchwindow:start_idx_matchwindow + window_size]
                match_window_pcd = random_lidar_embs[start_idx_matchwindow:start_idx_matchwindow + window_size]

                match_gt_imu = random_gt_imu_embs[start_idx_matchwindow:start_idx_matchwindow + window_size]
                match_gt_pcd = random_gt_lidar_embs[start_idx_matchwindow:start_idx_matchwindow + window_size]

                match_gt_imus.append(match_gt_imu)
                match_gt_pcds.append(match_gt_pcd)
                match_windows_imu.append(match_window_imu)
                match_windows_pcd.append(match_window_pcd)


        # Add to scene list
        augmented_scenes.append({
            tgt_modality : torch.stack(match_gt_pcds),
            src_modality : torch.stack(match_gt_imus),
            "%s_embs" % tgt_modality : torch.stack(match_windows_imu),
            "%s_embs" % src_modality : torch.stack(match_windows_pcd),
        })

    return augmented_scenes

def calculate_window_similarity(imu_window, lidar_windows):
    """
    Calculate similarity scores between an IMU window and all LiDAR windows.
    """
    similarities = []
    for lidar_window in lidar_windows:
        # Compute average similarity across the window
        frame_similarities = [
            1 - cosine(imu_emb, lidar_emb)
            for imu_emb, lidar_emb in zip(imu_window, lidar_window)
        ]
        similarities.append(sum(frame_similarities) / len(frame_similarities))
    return similarities

def match_imu_window_to_lidar(imu_window, lidar_windows):
    """
    Match an IMU window to the most similar LiDAR window.
    """
    similarities = calculate_window_similarity(imu_window, lidar_windows)
    best_match_index = similarities.index(max(similarities))
    return best_match_index, similarities

### Evaluation matching ACC over all augmented scenes
def eval_scenes(augmented_scenes, src_modality="IMU", tgt_modality="PC"):
    results = {}
    j = 0
    for scene in tqdm(augmented_scenes):
    #scene = augmented_scenes[0]

        ## for pcd visualization side-by-side...
        #full_seq = []
        #i = 0
        #for seq in scene[tgt_modality]:
        #    full_seq.append( seq + torch.Tensor([i,0,0]) )
        #    i += 1
        #seqs = torch.stack(full_seq)

        imu_embs = scene["%s_embs" % src_modality]
        pcd_embs = scene["%s_embs" % tgt_modality]

        predicted_matches = []
        sims = []
        ground_truth = np.arange(len(imu_embs))
        for imu_query in imu_embs:
            ### Find closest match.
            best_match_index, similarities = match_imu_window_to_lidar(imu_query, pcd_embs)
            predicted_matches.append(best_match_index)
            sims.append(similarities)

        results[j] = [predicted_matches, ground_truth]
        j += 1
    return results

