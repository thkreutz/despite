
import torch

class HMPEAR(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=True):
        self.augment = augment
        self.X = X
        self.y = y
    
    def augment_sequence(self, pc_sequence):
        pc_sequence = self.random_translation(pc_sequence)
        pc_sequence = self.add_gaussian_noise(pc_sequence)
        pc_sequence = self.random_scaling(pc_sequence)

        return pc_sequence

    def random_scaling(self, pc_sequence):
        # Generate a random scaling factor (1 + some uniform distribution)
        scale_factor = torch.FloatTensor(1).uniform_(0.7, 1.5).item()  # Example: Scale between 0.8 and 1.2
        # Scale the entire sequence
        return pc_sequence * scale_factor  # Apply scaling uniformly to all points
    
    def random_translation(self, pc_sequence):
        # Apply a single random translation to the entire sequence
        translation = torch.FloatTensor(3).uniform_(-0.3, 0.3)  # Adjust range as needed
        return pc_sequence + translation  # Apply to every point in the sequence

    def add_gaussian_noise(self, pc_sequence):
        # Add Gaussian noise to the entire point cloud sequence
        noise = torch.normal(mean=0, std=0.001, size=pc_sequence.size())  # Adjust standard deviation as needed
        return pc_sequence + noise
    
    def __getitem__(self, index):
        ### can randomize the number of frames as augmentation... but need to see how to handle the batches.
        #window_length_aug = np.random.randint(self.T-16, self.T) # minimum 16, max 32
        pc_seq = self.X[index]
        label = self.y[index]
        
        if self.augment:
            augmented_pc = self.augment_sequence(pc_seq)
            return augmented_pc.float(), label
        else:
            return pc_seq, label
        
    def __len__(self):
        return len(self.X)
