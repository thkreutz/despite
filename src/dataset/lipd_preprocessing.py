import numpy as np
import torch
import pickle
from tqdm import tqdm

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

def prepare_imu(imu_input, imu_acc):

    leftForeArm_imu, rightForeArm_imu,leftLeg_imu, rightLeg_imu,_,_ = imu_input.permute(2,0,1,3)
    leftForeArm_acc, rightForeArm_acc,leftLeg_acc, rightLeg_acc,_,_ = imu_acc.permute(2,0,1,3)
    B, T, _ = leftLeg_imu.shape
    imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 4*9)
        [
        leftLeg_imu.view(B, T, 9),
        rightLeg_imu.view(B, T, 9),
        leftForeArm_imu.view(B, T, 9),
        rightForeArm_imu.view(B, T, 9),
        leftLeg_acc.view(B, T, 3),
        rightLeg_acc.view(B, T, 3),
        leftForeArm_acc.view(B, T, 3),
        rightForeArm_acc.view(B, T, 3),
        ]
        , axis=2
    )
    
    return imu_input_cat

def load_lipd_data(m, datas, num_points=256):
    #m = "eTC"
    root_dataset_path = "/data/LIPD/LIPD"
    subjects = {}
    #num_points = 256
    for data in tqdm(datas):
        #print("???")
        if m == "eLIPD" or m =="LIPD_train":
            subject = data["pc"].split("/")[3] #eLIPD
            seq_id = data["pc"].split("/")[4] #eLIPD
        elif m in ["eTC", "eDIP", "ACCAD", "BMLmovi", "AIST", "CMU"]:
            subject = data["pc"].split("/")[4] #data["pc"].split("/")[3] eLIPD
            seq_id = data["pc"].split("/")[5] #data["pc"].split("/")[4] eLIPD

        pcd = np.fromfile(root_dataset_path + data["pc"][1:], dtype=np.float32).reshape(-1,3)
        if len(pcd)==0:
            pcd = np.array([[0,0,0]])
        pc_data = pcd - pcd.mean(0)
        pc_data = farthest_point_sample(pc_data, num_points)
        imu_acc = data["imu_acc"].reshape([-1,3])
        imu_ori = data["imu_ori"].reshape([-1,9])
        gt_joint = data["gt_joint"]
        gt = data["gt"]

        # if subject in dict already
        if subject in subjects:
            # if sequence in dict already
            if seq_id in subjects[subject]:
                # append to respective lists
                subjects[subject][seq_id]["pcd"].append(pc_data)
                subjects[subject][seq_id]["imu_acc"].append(imu_acc)
                subjects[subject][seq_id]["imu_ori"].append(imu_ori)
                subjects[subject][seq_id]["gt_joint"].append(gt_joint)
                subjects[subject][seq_id]["gt"].append(gt)
            else:
                # add new sequence to subject
                subjects[subject][seq_id] = {"pcd" : [pc_data], "imu_acc" : [imu_acc], "imu_ori" : [imu_ori],
                                            "gt_joint" : [gt_joint], "gt" : [gt]}
        else:
            # add new subject and at the same time new sequence
            subjects[subject] = {}
            subjects[subject][seq_id] = {"pcd" : [pc_data], "imu_acc" : [imu_acc], "imu_ori" : [imu_ori], "gt_joint" : [gt_joint], "gt" : [gt]}

    #### Transform everything to torch arrays
    test_subj_dataset = {}
    for subject in subjects.keys():
        if not subject in test_subj_dataset:
            test_subj_dataset[subject] = {}
        for seq in subjects[subject].keys():
            in_dict = {"PCD" : torch.Tensor(np.array(subjects[subject][seq]["pcd"])), 
                        "IMU" : prepare_imu(torch.Tensor(np.array(subjects[subject][seq]["imu_ori"])).unsqueeze(0), torch.Tensor(np.array(subjects[subject][seq]["imu_acc"])).unsqueeze(0)).squeeze(0),
                        "gt_joint" : torch.Tensor(np.array(subjects[subject][seq]["gt_joint"])),
                        "gt" : torch.Tensor(np.array(subjects[subject][seq]["gt"]))}
            
            test_subj_dataset[subject][seq] = in_dict

    return test_subj_dataset

def prepare_lipd(root_dataset_path = "/data/LIPD/LIPD/", num_points=256):
    ### Load each sequence
    testsets = ["eLIPD", "eTC", "eDIP"]
    trainsets = ["ACCAD", "BMLmovi", "LIPD_train", "AIST", "CMU"]

    sequence_datasets = {}
    for m in trainsets + testsets:
        ### TEST
        if m == 'eDIP':
            data_info_path = root_dataset_path + 'DIP_test.pkl'
        elif m=='eTC':
            data_info_path = root_dataset_path + 'TC_test.pkl'
        elif m =='eLIPD':
            data_info_path = root_dataset_path + 'LIPD_test.pkl'
        elif m == 'eLH':
            data_info_path = root_dataset_path + 'Test_lidarhuman.pkl'
            
        ### TRAIN
        elif m == "ACCAD":
            data_info_path = root_dataset_path + 'ACCAD.pkl'
        elif m == "BMLmovi":
            data_info_path = root_dataset_path + 'BMLmovi.pkl'
        elif m == "LIPD_train":
            data_info_path = root_dataset_path + 'LIPD_train.pkl'
        elif m == "AIST":
            data_info_path = root_dataset_path + 'AIST.pkl'
        elif m == "CMU":
            data_info_path = root_dataset_path + 'CMU.pkl'
        elif m == "Trans_train":
            data_info_path = root_dataset_path + 'Trans_train.pkl'

        file = open(data_info_path, 'rb')
        datas = pickle.load(file)
        file.close()
        
        print(m)
        #print(datas[0]["pc"])
        sequence_dataset = load_lipd_data(m=m, datas=datas, num_points=num_points)
        sequence_datasets[m] = sequence_dataset
        #test_subj_dataset = encode_all(datas, m)
        #test_subj_datasets[m] = test_subj_dataset

    return sequence_datasets

if __name__ == "__main__":
    root_dataset_path = "/data/LIPD/LIPD/" # set your path to LIPD dataset here 
    num_points = 256 # specificy number of LiDAR points here, we use 256 in the paper
    sequence_datasets = prepare_lipd()

    # save to pkl
    save_path = "/data/LIPD/LIPD_SEQUENCES_256p_TEST.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(sequence_datasets, f)