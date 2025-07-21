import numpy as np
import pickle as pkl
import torch
import glob
import os
from tqdm import tqdm

def normalize_pointcloud_sequence(pointcloud_sequence):
    """
    Normalize a sequence of point clouds to center at origin and scale to unit sphere.
    Args:
        pointcloud_sequence (list of np.ndarray): List of (N, 3) point clouds.
    Returns:
        list of np.ndarray: Normalized point cloud sequence.
    """
    centered_sequence = []
    for pc in pointcloud_sequence:
        centroid = np.mean(pc, axis=0)  # Compute centroid of the current frame
        centered_pc = pc - centroid    # Center the frame
        centered_sequence.append(centered_pc)
    
    return centered_sequence

def farthest_point_sample(xyz, npoint):
    ndataset = xyz.shape[0]
    if ndataset<npoint:
        repeat_n = int(npoint/ndataset)
        xyz = np.tile(xyz,(repeat_n,1))
        xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest =  np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

def preprocess_pointclouds_sliding_windows(point_clouds, window_length, stride=1):
    n_frames, n_points, dims = point_clouds.shape
    
    # Calculate the number of windows we can extract
    num_windows = n_frames - window_length + 1
    #num_windows = (n_frames - window_length) // stride + 1 if (n_frames - window_length) >= 0 else 0

    # Initialize a tensor to hold the sliding windows
    windows = []
    
    # Populate each window
    for i in range(num_windows):
        start_idx = i * stride
        windows.append(torch.Tensor(point_clouds[start_idx:start_idx + window_length]))
    
    return torch.stack(windows)

def rotate_pointcloud(pointcloud, axis, angle):
    """
    Rotate a 3D point cloud around a specific axis.
    
    Args:
        pointcloud: (N, 3) numpy array of 3D points.
        axis: 'x', 'y', or 'z' - the axis to rotate around.
        angle: Rotation angle in radians.
    
    Returns:
        Rotated pointcloud as (N, 3) numpy array.
    """
    # Rotation matrices
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Rotate point cloud
    rotated_pointcloud = pointcloud @ R.T
    return rotated_pointcloud

def rotate_pc(pc_hmpear_v):
    # rotate based on a random sampe. But definitialy we need to rotate x axis so they are not on top.
    angle_x = - np.pi / 2
    angle_z = + np.pi / 1.5
    pc_hmpear = rotate_pointcloud(pc_hmpear_v, axis='x', angle=angle_x)
    pc_hmpear = rotate_pointcloud(pc_hmpear, axis='z', angle=angle_z)
    return pc_hmpear

if __name__ == "__main__":
    window_length = 24


    ### TRAIN
    print("preprocessing HMPEAR train")
    with open('/data/HMPEAR/label/label/train_act.pkl', 'rb') as f:
        data = pkl.load(f)
    
    X_train = []
    X_train_clip_idxs = []
    y_train = []
    seq_names = []
    for i in tqdm(range(len(data))):
        seq_name = data[i]['seqname'] #: '0401_zjf_action_01'
        label = data[i]["action"]
        pcd_seq = data[i]["human_pc"]
        normalized_sequence = normalize_pointcloud_sequence(pcd_seq)
        
        fps_sequence = np.stack([rotate_pc(farthest_point_sample(frame, 1024)) for frame in normalized_sequence])
        
        # Get sliding windows of size 8
        if len(fps_sequence) >= window_length:
            sliding_windows = preprocess_pointclouds_sliding_windows(fps_sequence, window_length=window_length)
        
            #pcd_seq = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3])) for pcd in  normalized_sequence]
            X_train.append(sliding_windows)
            X_train_clip_idxs.append(seq_name)
            y_train.extend([label] * len(sliding_windows))
            seq_names.extend(seq_name)

    X_train = torch.vstack(X_train)
    torch.save({"PCD" : X_train, "labels" : y_train, "seq_names" : seq_names}, "/data/LIPD/hmpear_train_1024_24f.pt")

    ### TEST
    print("preprocessing HMPEAR train")
    with open('/data/HMPEAR/label/label/test_act.pkl', 'rb') as f:
        data = pkl.load(f)

    X_test = []
    y_test = []
    seq_names = []
    for i in tqdm(range(len(data))):
        seq_name = data[i]['seqname']
        label = data[i]["action"]
        pcd_seq = data[i]["human_pc"]
        normalized_sequence = normalize_pointcloud_sequence(pcd_seq)
        fps_sequence = np.stack([rotate_pc(farthest_point_sample(frame, 1024)) for frame in normalized_sequence])
        # Get sliding windows of size 8
        if len(fps_sequence) >= window_length:
            sliding_windows = preprocess_pointclouds_sliding_windows(fps_sequence, window_length=window_length)
        
            #pcd_seq = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3])) for pcd in  normalized_sequence]
            X_test.append(sliding_windows)
            y_test.extend([label] * len(sliding_windows))
            seq_names.extend(seq_name)

    X_test = torch.vstack(X_test)
    torch.save({"PCD" : X_test, "labels" : y_test, "seq_names" : seq_names}, "/data/LIPD/hmpear_test_1024_24f.pt")