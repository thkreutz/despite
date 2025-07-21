import torch
import numpy as np

def preprocess_pointclouds_sliding_windows_all(point_clouds, imu, smpl, labels, window_length, with_labels=True, stride=1):

    n_frames, n_points, dims = point_clouds.shape
    
    # Calculate the number of windows we can extract
    num_windows = n_frames - window_length + 1
    #num_windows = (n_frames - window_length) // stride + 1 if (n_frames - window_length) >= 0 else 0

    # Initialize a tensor to hold the sliding windows
    windows_pcd = []
    windows_imu = []
    windows_smpl = []
    Y = []
    # Populate each window
    for i in range(num_windows):
        start_idx = i * stride
        windows_pcd.append(point_clouds[start_idx:start_idx + window_length])
        windows_imu.append(imu[start_idx:start_idx + window_length])
        windows_smpl.append(smpl[start_idx:start_idx + window_length])
        Y.append(labels[start_idx + (window_length // 2)])
    #print(len(windows))
    #print(windows[0])
    return torch.stack(windows_pcd), torch.stack(windows_imu), torch.stack(windows_smpl), Y


def preprocess_pointclouds_sliding_windows_all_babel(point_clouds, imu, smpl, labels, window_length, action_to_idx_label, with_labels=True, stride=1):

    n_frames, n_points, dims = point_clouds.shape
    
    # Calculate the number of windows we can extract
    num_windows = n_frames - window_length + 1
    #num_windows = (n_frames - window_length) // stride + 1 if (n_frames - window_length) >= 0 else 0

    # Initialize a tensor to hold the sliding windows
    windows_pcd = []
    windows_imu = []
    windows_smpl = []
    Y = []
    # Populate each window
    for i in range(num_windows):
        start_idx = i * stride

        # filter if it is in babel text split
        #lbl = labels[start_idx + window_length // 2]
        #print(lbl)
        #if lbl in action_to_idx_label:
        Y.append(labels[start_idx + (window_length // 2)])
        windows_pcd.append(point_clouds[start_idx:start_idx + window_length])
        windows_imu.append(imu[start_idx:start_idx + window_length])
        windows_smpl.append(smpl[start_idx:start_idx + window_length])
        
    #print(len(windows))
    #print(windows[0])
    return torch.stack(windows_pcd), torch.stack(windows_imu), torch.stack(windows_smpl), Y

def preprocess_pointclouds_sliding_windows(point_clouds, labels, window_length, with_labels=True, stride=1):
    n_frames, n_points, dims = point_clouds.shape
    
    # Calculate the number of windows we can extract
    num_windows = n_frames - window_length + 1
    #num_windows = (n_frames - window_length) // stride + 1 if (n_frames - window_length) >= 0 else 0

    # Initialize a tensor to hold the sliding windows
    windows = []
    Y = []
    # Populate each window
    for i in range(num_windows):
        start_idx = i * stride
        windows.append(point_clouds[start_idx:start_idx + window_length])
        Y.append(labels[start_idx + (window_length // 2)])
    
    #print(len(windows))
    #print(windows[0])
    return torch.stack(windows), Y

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
