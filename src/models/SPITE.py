import torch
import torch
import torch.nn as nn
import inspect

def get_name(modalities, with_generator):
    order = ["skeleton", "pc", "imu", "text"]
    mapping = {"skeleton": "S", "pc": "P", "imu": "I", "text": "T"}
    
    sorted_modalities = sorted(modalities, key=lambda x: order.index(x))
    result = "".join(mapping[m] for m in sorted_modalities) + "E"
    #print(modalities)
    #print(result)
    if with_generator:
        return result + "Gen"
    else:
        return result



def get_binder_class(modalities, with_generator):
    class_name = get_name(modalities, with_generator) + "_BINDER"
    #class_name = class_name + "Gen" if with_generator else class_name 
    return globals().get(class_name)

def instantiate_binder(modalities, with_generator, imu_encoder=None, pointcloud_encoder=None, skeleton_encoder=None, skeleton_generator=None):
    binder_class = get_binder_class(modalities, with_generator)
    print("Loading class", binder_class)
    if binder_class is None:
        raise ValueError(f"Binder class {binder_class}_BINDER not found")
    
    init_params = inspect.signature(binder_class.__init__).parameters
    available_kwargs = {
        "imu_encoder": imu_encoder,
        "pointcloud_encoder": pointcloud_encoder,
        "skeleton_encoder": skeleton_encoder,
        "skeleton_generator": skeleton_generator
    }
    
    filtered_kwargs = {k: v for k, v in available_kwargs.items() if k in init_params and v is not None}
    
    return binder_class(**filtered_kwargs)


def instantiate_binder_class_from_name(name, imu_encoder=None, pointcloud_encoder=None, skeleton_encoder=None, skeleton_generator=None):
    binder_class = globals().get(name + "_BINDER")
    print("Loading class", binder_class)
    if binder_class is None:
        raise ValueError(f"Binder class {binder_class}_BINDER not found")
    
    init_params = inspect.signature(binder_class.__init__).parameters
    available_kwargs = {
        "imu_encoder": imu_encoder,
        "pointcloud_encoder": pointcloud_encoder,
        "skeleton_encoder": skeleton_encoder,
        "skeleton_generator": skeleton_generator
    }
    
    filtered_kwargs = {k: v for k, v in available_kwargs.items() if k in init_params and v is not None}
    
    return binder_class(**filtered_kwargs)


# Skeleton-Pointcloud-IMU
class SPITEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_pc, batch_skeleton, batch_text, with_text=None):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z
        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        # Preprocessing moved to skeleton encoder.
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        ### gen expects a batch z, y, mask (from mask the lenght comes)
        
        # Reconstruct the skeleton
        seq_length = batch_pc.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)

        return out

class SPITE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_imu, batch_pc, batch_skeleton):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        out["pc"] = self.pointcloud_encoder(batch_pc)
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

class SPIEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_pc, batch_skeleton):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z
        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        # Preprocessing moved to skeleton encoder.
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        ### gen expects a batch z, y, mask (from mask the lenght comes)
        
        # Reconstruct the skeleton
        seq_length = batch_pc.shape[1]
 
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)

        return out

class SPIE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_imu, batch_pc, batch_skeleton):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        out["pc"] = self.pointcloud_encoder(batch_pc)

        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out


# IMU-Pointcloud
class PITEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_pc, batch_text, with_text=None):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z
        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        seq_length = batch_pc.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        return out

class PITE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder


    def forward(self, batch_imu, batch_pc):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        out["pc"] = self.pointcloud_encoder(batch_pc)
        return out

class PIEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_pc):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z
        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        seq_length = batch_pc.shape[1]
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        return out

class PIE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, pointcloud_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.pointcloud_encoder = pointcloud_encoder


    def forward(self, batch_imu, batch_pc):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        out["pc"] = self.pointcloud_encoder(batch_pc)
        return out

# Skeleton-Pointcloud
class SPTEGen_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_pc, batch_skeleton, batch_text, with_text=None):
        out = {}

        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        # Preprocessing moved to skeleton encoder.
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        ### gen expects a batch z, y, mask (from mask the lenght comes)
        
        # Reconstruct the skeleton
        seq_length = batch_pc.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)

        return out

class SPTE_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder, skeleton_encoder):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_pc, batch_skeleton):
        out = {}
        out["pc"] = self.pointcloud_encoder(batch_pc)
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

class SPEGen_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_pc, batch_skeleton):
        out = {}

        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        # Preprocessing moved to skeleton encoder.
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        ### gen expects a batch z, y, mask (from mask the lenght comes)
        
        # Reconstruct the skeleton
        seq_length = batch_pc.shape[1]
        
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)

        return out

class SPE_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder, skeleton_encoder):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_pc, batch_skeleton):
        out = {}
        out["pc"] = self.pointcloud_encoder(batch_pc)
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

# Skeleton-IMU
class SITE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, skeleton_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_imu, batch_skeleton):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

class SITEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_skeleton, batch_text, with_text=None):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z

        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z

        seq_length = batch_skeleton.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        return out

class SIE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, skeleton_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_imu, batch_skeleton):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

class SIEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_skeleton):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z

        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z

        seq_length = batch_skeleton.shape[1]
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        return out

# Skeleton-Text
class STE_BINDER(torch.nn.Module):
    
    def __init__(self, skeleton_encoder):
        super().__init__()
        self.skeleton_encoder = skeleton_encoder


    def forward(self, batch_skeleton):
        out = {}
        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z
        return out

class STEGen_BINDER(torch.nn.Module):
    
    def __init__(self, skeleton_encoder, skeleton_generator):
        super().__init__()
        self.skeleton_encoder = skeleton_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_skeleton, batch_text, with_text=None):
        out = {}

        skeleton_z = self.skeleton_encoder(batch_skeleton)["mu"]
        out["skeleton"] = skeleton_z

        seq_length = batch_skeleton.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_skeleton"] = self.skeleton_generator({  "z": skeleton_z,
                                                "y" : torch.zeros(batch_skeleton.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_skeleton.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        return out

# IMU-Text
class ITE_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder):
        super().__init__()
        self.imu_encoder = imu_encoder


    def forward(self, batch_imu):
        out = {}
        out["imu"] = self.imu_encoder(batch_imu)
        return out

class ITEGen_BINDER(torch.nn.Module):
    
    def __init__(self, imu_encoder, skeleton_generator):
        super().__init__()
        self.imu_encoder = imu_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_imu, batch_text, with_text=None):
        out = {}
        imu_z = self.imu_encoder(batch_imu)
        out["imu"] = imu_z


        seq_length = batch_imu.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        out["gen_imu"] = self.skeleton_generator({  "z": imu_z,
                                                "y" : torch.zeros(batch_imu.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_imu.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        
        return out

# Pointcloud-Text
class PTEGen_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder, skeleton_generator):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder

        # use the same generator for all different modality representations => auxilary loss to push them together because they should generate the same skeleton
        self.skeleton_generator = skeleton_generator

    def forward(self, batch_pc, batch_text, with_text=None):
        out = {}

        pc_z = self.pointcloud_encoder(batch_pc)
        out["pc"] = pc_z

        # Reconstruct the skeleton
        seq_length = batch_pc.shape[1]
        if with_text:
            out["gen_text"] = self.skeleton_generator({  "z": batch_text,
                                                    "y" : torch.zeros(batch_text.shape[0]).long().to("cuda"), 
                                                    "mask" :  torch.ones((batch_text.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)
        out["gen_pc"] = self.skeleton_generator({  "z": pc_z,
                                                "y" : torch.zeros(batch_pc.shape[0]).long().to("cuda"), 
                                                "mask" :  torch.ones((batch_pc.shape[0], seq_length), dtype=torch.long).long().to("cuda")}).permute(0, 3, 1, 2)

        return out

class PTE_BINDER(torch.nn.Module):
    
    def __init__(self, pointcloud_encoder):
        super().__init__()
        self.pointcloud_encoder = pointcloud_encoder


    def forward(self, batch_pc):
        out = {}
        out["pc"] = self.pointcloud_encoder(batch_pc)
        return out


if __name__ == "__main__":
    ## test
    embed_dim = 128
    num_joints = 24 # keep this the same because we have only one dataset.
    n_feats = 3 # keep this the same because we have only one dataset.

    ## All the combinations
    modalities_tests = [ ["imu", "text"], ["pc", "text"], ["skeleton", "text"], 
                        ["imu", "pc"], ["imu", "skeleton"], ["pc", "skeleton"],  ["pc", "imu"],
                        ["imu", "pc", "text"], ["imu", "skeleton", "text"], ["pc", "skeleton", "text"], ["pc", "imu", "text"],
                        ["imu", "pc", "skeleton"],
                        ["imu", "pc", "skeleton", "text"]
                        ]

    for with_generator in [0,1]:
        for modalities in modalities_tests:
            print(modalities, with_generator)
            skeleton = model_loader.load_skeleton_encoder(embed_dim, num_joints, n_feats, device="cpu") if "skeleton" in modalities else None
            imu = model_loader.load_imu_encoder(embed_dim, device="cpu") if "imu" in modalities else None
            pc = model_loader.load_pst_transformer(embed_dim, device="cpu") if "pc" in modalities else None
            skeleton_gen = model_loader.load_skeleton_generator(embed_dim, num_joints, n_feats, device="cpu") if with_generator else None

            binder = SPITE.instantiate_binder(modalities, with_generator, imu, pc, skeleton, skeleton_gen)
