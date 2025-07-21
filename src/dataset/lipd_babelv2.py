##### V2 has Babel Subset Train-Testsplit
import torch
from src.dataset import sliding_window_utils
from pandas.io.json._normalize import nested_to_record
import numpy as np
from tqdm import tqdm
from src.dataset import babel

#### For representation learning
class LIPDBabelv2(torch.utils.data.Dataset):
    def __init__(self, sequences, sequences_babel_train, sequences_babel_val, num_frames=24, augment=False, train=True, modalities=["pc", "imu", "skeleton", "text"]):
        
        ## Here we train on all and only leave out the test set of Babel
        sets = ["ACCAD", "BMLmovi", "CMU", "eTC", "LIPD_train", "AIST", "eLIPD", "eDIP"]

        #if train:
        #    
        #else:
        #    sets = []
        self.T = num_frames
        ### Train on all data we can get from LIPD dataset.
        ### They have real and synthetic LiDAR-IMU data 
        ### where all IMUS are supposed to be on the same joint positions.
        self.sequences = {k : sequences[k] for k in sets}
        self.dataset = nested_to_record(self.sequences, sep="/", max_level=2)
        self.dataset = {k : v for k,v in self.dataset.items() if len(v["PCD"]) >= num_frames+1}
        # merge the babel text labels
        if train:
            self.sequences_babel = {k : sequences_babel_train[k] for k in ["ACCAD", "BMLmovi", "CMU", "eTC"]}
            self.sequences_babel = nested_to_record(self.sequences_babel, sep="/", max_level=1)
        else:
            self.sequences_babel = {k : sequences_babel_val[k] for k in ["ACCAD", "BMLmovi", "CMU", "eTC"]}
            self.sequences_babel = nested_to_record(self.sequences_babel, sep="/", max_level=1)


        ### In this way, we can evalute training on babel training set and 

        # Preprocess into sliding windows
        # each k in self.dataset is a sequence.
        self.X_pcd = []
        self.X_imu = []
        self.X_smpl = []
        self.X_text = []


        #print(len(self.dataset))
        #self.missings = []
        #count_missing = 0
        for k, v in self.dataset.items():
            

            #### Filter out the sequences from ACCAD, BMLmovi, CMU, and TC that are in train and val set.

            windows_text = ["human activity"] # a general text embedding for all motions, we dont count them during training.
            if k in self.sequences_babel:

                text_label_type = ""
                if train:
                    text_label_type = "raw_text" ## train with specific labels
                else:
                    text_label_type = "action_cat" ## test with the categories for retrieval.

                windows_pcd, windows_imu, windows_smpl, windows_text = sliding_window_utils.preprocess_pointclouds_sliding_windows_all(v["PCD"], 
                                v["IMU"], v["gt_joint"].reshape(-1, 24, 3), self.sequences_babel[k][text_label_type], num_frames, with_labels=True, stride=1)
                
                self.X_text.extend(windows_text)
                self.X_pcd.append(windows_pcd)
                self.X_imu.append(windows_imu)
                self.X_smpl.append(windows_smpl)
            else:
                
                # Only add if not in babel val/train set since its not in our current babel split.

                if train:
                    if k.startswith("LIPD_train") or k.startswith("eLIPD") or k.startswith("AIST") or k.startswith("eDIP"):
                        windows_pcd, windows_imu, windows_smpl = sliding_window_utils.preprocess_sliding_windows(v["PCD"], v["IMU"], v["gt_joint"].reshape(-1, 24, 3), self.T, stride=1)

                        self.X_text.extend(windows_text * len(windows_pcd))
                        self.X_pcd.append(windows_pcd)
                        self.X_imu.append(windows_imu)
                        self.X_smpl.append(windows_smpl)
                    
                    ### There are a bunch of sequences that are in lipd but not in the babel val or train split. 
                    ### => they are instead in the test split. so thats why there are a smaller number of sequences in the training/val set in lipd-babelv2.
                    #else:
                    #    self.missings.append(k)
                    #    count_missing += 1
        
        #print(count_missing)
        #self.missings])
                
                #else: # Keep testset of LIPD for skeleton generation testing
                #    if k.startswith("eLIPD"):
                #        windows_pcd, windows_imu, windows_smpl = preprocess_sliding_windows(v["PCD"], v["IMU"], v["gt_joint"].reshape(-1, 24, 3), self.T, stride=1)

                #        self.X_text.extend(windows_text * len(windows_pcd))
                #        self.X_pcd.append(windows_pcd)
                #        self.X_imu.append(windows_imu)
                #        self.X_smpl.append(windows_smpl)
                
        self.X_pcd = torch.vstack(self.X_pcd)
        self.X_imu = torch.vstack(self.X_imu)
        self.X_smpl = torch.vstack(self.X_smpl)

        self.dataset_indices = list(self.dataset.keys())
        self.augment = augment

        self.modalities = modalities
        
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


### For classification
class LIPDBabelv2CLS(torch.utils.data.Dataset):
    def __init__(self, sequences_babel_train, sequences_babel_val, babel_v=120, num_frames=24, augment=False, train=True, modalities=["pc", "imu", "skeleton", "text"]):
        # in both subsets...
        sets = ["ACCAD", "BMLmovi", "CMU", "eTC"]
        #if train:
        #    
        #else:
        #    sets = []
        self.T = num_frames
        ### Train on all data we can get from LIPD dataset.
        ### They have real and synthetic LiDAR-IMU data 
        ### where all IMUS are supposed to be on the same joint positions.

        # merge the babel text labels
        if train:
            sequences_babel = {k : sequences_babel_train[k] for k in ["ACCAD", "BMLmovi", "CMU", "eTC"]}
            sequences_babel = nested_to_record(sequences_babel, sep="/", max_level=1)
        else:
            sequences_babel = {k : sequences_babel_val[k] for k in ["ACCAD", "BMLmovi", "CMU", "eTC"]}
            sequences_babel = nested_to_record(sequences_babel, sep="/", max_level=1)


        ### In this way, we can evalute training on babel training set and 

        # Preprocess into sliding windows
        # each k in self.dataset is a sequence.
        self.X_pcd = []
        self.X_imu = []
        self.X_smpl = []
        self.X_text = []
        # for babel 60
        self.action_to_idx_label = { a: i for a, i in babel.action_label_to_idx.items() if i < babel_v}

        for k, v in sequences_babel.items():

            text_label_type = "raw_text"
            #if train:
            #    text_label_type = "raw_text" ## train with specific labels
            #else:
            #    text_label_type = "action_cat" ## test with the categories for retrieval.
            if len(v["PCD"]) >= self.T:
                #print(len(v["PCD"]))
                windows_pcd, windows_imu, windows_smpl, windows_text = sliding_window_utils.preprocess_pointclouds_sliding_windows_all_babel(v["PCD"], 
                            v["IMU"], v["gt_joint"].reshape(-1, 24, 3), v[text_label_type], num_frames, self.action_to_idx_label, with_labels=True, stride=1)

                self.X_text.extend(windows_text)
                self.X_pcd.append(windows_pcd)
                self.X_imu.append(windows_imu)
                self.X_smpl.append(windows_smpl)
        
        # Now filter for the labels.

         
        self.X_pcd = torch.vstack(self.X_pcd)
        self.X_imu = torch.vstack(self.X_imu)
        self.X_smpl = torch.vstack(self.X_smpl)
        #self.X_text = torch.vstack(self.X_text)

        mask_keep = []
        self.Y = []
        for i in range(len(self.X_pcd)):
            if self.X_text[i] in babel.in_both_subsets:
                mask_keep.append(i)
                self.Y.append(babel.action_label_to_idx[self.X_text[i]])
        
        mask_keep = np.array(mask_keep)

        self.X_pcd = self.X_pcd[mask_keep]
        self.X_imu = self.X_imu[mask_keep]
        self.X_smpl = self.X_smpl[mask_keep]
        self.X_text = np.array(self.X_text)[mask_keep]

        assert(len(self.Y) == len(self.X_pcd))

        self.augment = augment
        self.modalities = modalities

        
        
    def __getitem__(self, index):
        
        #text_seq = babel.action_label_to_idx[self.X_text[index]] # map to babel
        
        data = {}
        if "pc" in self.modalities:
            data["batch_pc"] = self.X_pcd[index]
        if "imu" in self.modalities:
            data["batch_imu"] = self.X_imu[index]
        if "skeleton" in self.modalities:
            data["batch_skeleton"] = self.X_smpl[index]
        #if "text" in self.modalities:
        data["batch_label"] = self.action_to_idx_label[self.X_text[index]]
            #data["text_mask"] = int(data["batch_text"] != "human activity")  # Label for annotation masking

        if self.augment:
            if "pc" in data:
                data["batch_pc"] = self.augment_pc_sequence(data["batch_pc"]).float()
            if "imu" in data:
                data["batch_imu"] = self.augment_imu(data["batch_imu"]).float()
            if "skeleton" in data:
                data["batch_skeleton"] = self.augment_joints_sequence(data["batch_skeleton"]).float()

        return data["batch_%s" % self.modalities[0]], data["batch_label"]
    
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

