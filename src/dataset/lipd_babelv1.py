##### V1 has the official train-testsplit
import torch
from src.dataset import sliding_window_utils
from pandas.io.json._normalize import nested_to_record
import numpy as np
from tqdm import tqdm


class LIPDBabelv1(torch.utils.data.Dataset):
    def __init__(self, sequences, sequences_babel_train, sequences_babel_val, num_frames=24, augment=True, train=True, modalities=["pc", "imu", "skeleton", "text"]):
        

        #### Testing occurs in separate modules so we dont explicitly do a train-test differentiaten here.

        if train:
            sets = ["ACCAD", "BMLmovi", "LIPD_train", "AIST", "CMU"]
        else:
            sets = ["eLIPD", "eTC", "eDIP"]

        self.T = num_frames
        ### Train on all data we can get from LIPD dataset.
        ### They have real and synthetic LiDAR-IMU data 
        ### where all IMUS are supposed to be on the same joint positions.
        self.sequences = {k : sequences[k] for k in sets}
        self.dataset = nested_to_record(self.sequences, sep="/", max_level=2)
        self.dataset = {k : v for k,v in self.dataset.items() if len(v["PCD"]) >= num_frames+1}

        # merge the babel text labels
        sequences_babel_a = {k : sequences_babel_train[k] for k in ["ACCAD", "BMLmovi", "CMU"]}
        sequences_babel_b = {k : sequences_babel_val[k] for k in ["ACCAD", "BMLmovi", "CMU"]}
        sequences_babel_a = nested_to_record(sequences_babel_a, sep="/", max_level=1)
        sequences_babel_b = nested_to_record(sequences_babel_b, sep="/", max_level=1)
        sequences_babel = {**sequences_babel_a, **sequences_babel_b}

        ## includes both subsets.
        ## includes only the training part. from lipd.

        # Preprocess into sliding windows
        # each k in self.dataset is a sequence.
        self.X_pcd = []
        self.X_imu = []
        self.X_smpl = []
        self.X_text = []

        for k, v in self.dataset.items():
            
            windows_text = ["human activity"] # a general text embedding for all motions.
            if k in sequences_babel:
                windows_pcd, windows_imu, windows_smpl, windows_text = sliding_window_utils.preprocess_pointclouds_sliding_windows_all(v["PCD"], 
                                v["IMU"], v["gt_joint"].reshape(-1, 24, 3), sequences_babel[k]["raw_text"], num_frames, with_labels=True, stride=1)
                self.X_text.extend(windows_text)
            else:
                windows_pcd, windows_imu, windows_smpl = sliding_window_utils.preprocess_sliding_windows(v["PCD"], v["IMU"], v["gt_joint"].reshape(-1, 24, 3), self.T, stride=1)
                self.X_text.extend(windows_text * len(windows_pcd))
            self.X_pcd.append(windows_pcd)
            self.X_imu.append(windows_imu)
            self.X_smpl.append(windows_smpl)
                
        self.X_pcd = torch.vstack(self.X_pcd)
        self.X_imu = torch.vstack(self.X_imu)
        self.X_smpl = torch.vstack(self.X_smpl)
        #self.X_text = torch.vstack(self.X_text)

        self.dataset_indices = list(self.dataset.keys())
        self.augment = augment

        self.modalities = modalities # Preprocess all modalities, but during training only retrieve relevant ones.
        
    def __getitem__(self, index):
        data = {}
        if "pc" in self.modalities:
            data["batch_pc"] = self.X_pcd[index]
        if "imu" in self.modalities:
            data["batch_imu"] = self.X_imu[index]

        # => Always return skeleton because we have a generator
        #if "skeleton" in self.modalities:
        data["batch_skeleton"] = self.X_smpl[index]
        if "text" in self.modalities:
            data["batch_text"] = self.X_text[index]
            data["text_mask"] = int(data["batch_text"] != "human activity")  # Label for annotation masking

        if self.augment:
            if "pc" in data:
                data["batch_pc"] = self.augment_pc_sequence(data["batch_pc"]).float()
            if "imu" in data:
                data["batch_imu"] = self.augment_imu(data["batch_imu"]).float()
            data["batch_skeleton"] = self.augment_joints_sequence(data["batch_skeleton"]).float()

        return data
    
    def augment_pc_sequence(self, pc_sequence):
        rnd_scale_factor = torch.FloatTensor(1).uniform_(0.7, 1.5)
        rnd_translation = torch.FloatTensor(3).uniform_(-0.3, 0.3)

        pc_sequence = self.random_translation(pc_sequence, rnd_translation)
        pc_sequence = self.add_gaussian_noise(pc_sequence)
        pc_sequence = self.random_scaling(pc_sequence, rnd_scale_factor)

        return pc_sequence

    def augment_joints_sequence(self, joints_sequence):
        rnd_scale_factor = torch.FloatTensor(1).uniform_(0.7, 1.5)
        rnd_translation = torch.FloatTensor(3).uniform_(-0.3, 0.3)

        joints_sequence = self.random_translation_joints(joints_sequence, rnd_translation)
        joints_sequence = self.add_gaussian_noise_joints(joints_sequence)
        joints_sequence = self.random_scaling_joints(joints_sequence, rnd_scale_factor)
        return joints_sequence

    def random_scaling(self, pc_sequence, scale_factor):
        # Generate a random scaling factor (1 + some uniform distribution)
        #scale_factor = torch.FloatTensor(1).uniform_(0.7, 1.5).item()  # Example: Scale between 0.8 and 1.2
        # Scale the entire sequence
        return pc_sequence * scale_factor  # Apply scaling uniformly to all points
    
    def random_translation(self, pc_sequence, translation):
        # Apply a single random translation to the entire sequence
        #translation = torch.FloatTensor(3).uniform_(-0.3, 0.3)  # Adjust range as needed
        return pc_sequence + translation  # Apply to every point in the sequence

    def add_gaussian_noise(self, pc_sequence):
        # Add Gaussian noise to the entire point cloud sequence
        noise = torch.normal(mean=0, std=0.005, size=pc_sequence.size())  # Adjust standard deviation as needed
        return pc_sequence + noise
    
    def add_jitter(self, data, sigma=0.01):
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise
    
    def augment_imu(self, imu):
        return self.scale(self.add_jitter(imu))
    
    def scale(self, data, factor_range=(0.95, 1.05)):
        factor = np.random.uniform(*factor_range)
        return data * factor

    def random_scaling_joints(self, joints_sequence, scale_factor, scale_range=(0.8, 1.2)):
        #scale_factor = torch.FloatTensor(1).uniform_(*scale_range).item()
        return joints_sequence * scale_factor
    
    def random_translation_joints(self, joints_sequence, translation, translation_range=(-0.1, 0.1)):
        #translation = torch.FloatTensor(1, 3).uniform_(*translation_range)
        return joints_sequence + translation
    
    def add_gaussian_noise_joints(self, joints_sequence, noise_std=0.005):
        noise = torch.normal(mean=0, std=noise_std, size=joints_sequence.size())
        return joints_sequence + noise

    def __len__(self):
        return len(self.X_pcd)

