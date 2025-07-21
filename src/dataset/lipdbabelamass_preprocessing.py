import json
from os.path import join as ospj
import numpy as np
import pickle
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch
from tqdm import tqdm
import os
from pandas.io.json._normalize import nested_to_record
from os.path import join as ospj

def load_babel():  
    d_folder = '/data/BABEL/babel_v1.0_release'  # Data folder
    l_babel_dense_files = ['train', 'val', 'test']
    l_babel_extra_files = ['extra_train', 'extra_val']

    # BABEL Dataset 
    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file+'.json')))
        
    for file in l_babel_extra_files:
        babel[file] = json.load(open(ospj(d_folder, file+'.json')))    

    ### Filter babel

    return babel


def load_LIPD(dataset_path):
    ## AMASS = ACCAD, BMLMovi, CMU, TotalCapture
    file = open(dataset_path,'rb')
    datas = pickle.load(file)
    return datas

def amases_pose_to_joints(bdata, body_model, joints_to_use, comp_device="cpu"):
    comp_device = "cpu"
    body_params = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    }

    body_motion = body_model(pose_body=body_params["pose_body"], pose_hand=body_params["pose_hand"], root_orient=body_params["root_orient"])
    joints = c2c(body_motion.Jtr)

    print(joints.shape)
    return joints[:, joints_to_use]
    
def amass_poses_and_trans(bdata, body_model, joints_to_use):
    #joints_to_use = np.array([0,1,2,3,4,5,6,7,8,9,10, 11, 12,13,14, 15, 16,17,18,19,20,21,22,37])

    comp_device = "cpu"
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    time_length = len(bdata['trans'])
    body_params = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }

    body_motion = body_model(pose_body=body_params["pose_body"], pose_hand=body_params["pose_hand"], root_orient=body_params["root_orient"])
    joints = c2c(body_motion.Jtr)

    return joints[:, joints_to_use]


def lipd_to_amass_id(lipd_id, m, data_root_path="/data/AMASS/"):
    #data_root_path = "/data/AMASS/"

    if m == "ACCAD":
        ## For ACCAD
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("_", " ")
        amass_id = amass_id.replace(" stageii", "_poses.npz")
        return os.path.join(data_root_path, m, amass_pre, amass_id)

    elif m == "eTC":
        ## BMLmovi, CMU, TC
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("stageii", "poses.npz")
        return os.path.join(data_root_path, "TotalCapture", amass_pre, amass_id)
    else: 
        ## BMLmovi, CMU
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("stageii", "poses.npz")
        return os.path.join(data_root_path, m, amass_pre, amass_id)

def lipd_to_babel(lipd_id, m, data_root_path=""):
    #data_root_path = "/data/AMASS/"

    if m == "ACCAD":
        ## For ACCAD
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("_", " ")
        amass_id = amass_id.replace(" stageii", "_poses.npz")
        return os.path.join(m, m, amass_pre, amass_id)

    elif m == "eTC":
        ## BMLmovi, CMU, TC
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("stageii", "poses.npz")
        return os.path.join("TotalCapture", "TotalCapture", amass_pre, amass_id)
    else: 
        ## BMLmovi, CMU
        amass_pre = lipd_id.split("/")[0]
        amass_id = lipd_id.split("/")[1]
        amass_id = amass_id.replace("stageii", "poses.npz")
        return os.path.join(m, m, amass_pre, amass_id)

def get_babel_labels(data, babel_dict, fps, target_fps=10):
    ### This method is based on MotionClip https://github.com/GuyTevet/MotionCLIP/src/datasets/amass_parser.py
    # Seq. labels
    seq_raw_labels, seq_proc_label, seq_act_cat = [], [], []
    frame_raw_text_labels = np.full(data['poses'].shape[0], "", dtype=object)
    frame_proc_text_labels = np.full(data['poses'].shape[0], "", dtype=object)
    frame_action_cat = np.full(data['poses'].shape[0], "", dtype=object)

    for label_dict in babel_dict['seq_ann']['labels']:
        seq_raw_labels.extend([label_dict['raw_label']])
        seq_proc_label.extend([label_dict['proc_label']])
        if label_dict['act_cat'] is not None:
            seq_act_cat.extend(label_dict['act_cat'])

    # Frames labels
    #print(seq_raw_labels)
    if babel_dict['frame_ann'] is None:
        
        frame_raw_labels = "and ".join(seq_raw_labels)
        frame_proc_labels = "and ".join(seq_proc_label)
        start_frame = 0
        end_frame = data['poses'].shape[0]
        frame_raw_text_labels[start_frame:end_frame] = frame_raw_labels
        frame_proc_text_labels[start_frame:end_frame] = frame_proc_labels
        frame_action_cat[start_frame:end_frame] = ",".join(seq_act_cat)
    else:
        for label_dict in babel_dict['frame_ann']['labels']:
            start_frame = round(label_dict['start_t'] * fps)
            end_frame = round(label_dict['end_t'] * fps)
            frame_raw_text_labels[start_frame:end_frame] = label_dict['raw_label']
            frame_proc_text_labels[start_frame:end_frame] = label_dict['proc_label']
            if label_dict['act_cat'] is not None:
                frame_action_cat[start_frame:end_frame] = str(",".join(label_dict['act_cat']))
    max_fps_dist = 5
    if target_fps is not None:
        mocap_framerate = float(data['mocap_framerate'])
        sampling_freq = round(mocap_framerate / target_fps)
        if abs(mocap_framerate / float(sampling_freq) - target_fps) > max_fps_dist:
            print('Will not sample [{}]fps seq with sampling_freq [{}], since target_fps=[{}], max_fps_dist=[{}]'
                 .format(mocap_framerate, sampling_freq, target_fps, max_fps_dist))
        #    continue
        # pose = data['poses'][:, joints_to_use]
        pose = data['poses'][0::sampling_freq]
        frame_raw_text_labels = frame_raw_text_labels[0::sampling_freq]
        frame_proc_text_labels = frame_proc_text_labels[0::sampling_freq]
        frame_action_cat = frame_action_cat[0::sampling_freq]
    return pose, frame_raw_text_labels, frame_proc_text_labels, frame_action_cat
    
def get_babel_labels_and_amass_poses(seq_dataset, m, babel, babel_split="train", amass_data_root_path="/data/AMASS/", target_fps=10):

    # get mapping from amass paths to ids in babel dict

    #feat_ps = {}
    #for ts in ["train", "val", "test"]:
    #    for bsid in babel[ts].keys():
    #        feat_ps[babel[ts][bsid]["feat_p"]] = bsid

    feat_ps = {babel[babel_split][bsid]["feat_p"] : bsid for bsid in babel[babel_split].keys()}

    # get unique ids for each sequence in lipd subset dataset="m" (e.g., m = ACCAD)
    lipd = seq_dataset[m]
    lipd = nested_to_record(lipd, sep="/", max_level=1)

    # Make a mapping from the AMASS paths to babel_ids (allows to connect LIPD to Babel)
    if m == "eTC":
        m_bab = "TotalCapture"
    else:
        m_bab = m

    # Make a mapping from the AMASS paths to babel_ids (allows to connect LIPD to Babel) for the specific dataset only
    path_to_babel_id = {a : b for a, b in feat_ps.items() if m_bab in a}

    # get the mappings from lipd to babel dict and lipd to amass sequences
    lipd_in_babel = {}
    lipd_in_amass = {}
    
    for lipd_id in lipd.keys():
        babel_path = lipd_to_babel(lipd_id, m, data_root_path=amass_data_root_path)
        amass_path = lipd_to_amass_id(lipd_id, m, data_root_path=amass_data_root_path)
        # babel seems to miss some amass sequences, so we filter here if they actually exist.
        if babel_path in path_to_babel_id:
            lipd_in_babel[lipd_id] = path_to_babel_id[babel_path]
            lipd_in_amass[lipd_id] = amass_path


    # get the labels from babel downsampled to n fps lipd for each amass sequence.
    amass_poses = {}
    raw_text_labels = {}
    proc_text_labels = {}
    action_cat_labels = {}

    lipd_valid = [] # store the ids where we actually add labels because they match the sequence length.
    for lipd_id, amass_path in tqdm(lipd_in_amass.items()):
        babel_dict = babel[babel_split][lipd_in_babel[lipd_id]]
        data = np.load(amass_path)
        duration_t = babel_dict['dur']
        fps = data['poses'].shape[0] / duration_t
        poses, raw_text_lbl, proc_text_lbl, action_cat_lbl = get_babel_labels(data, babel_dict, fps, target_fps=target_fps)
        amass_poses[lipd_id] = poses
        raw_text_labels[lipd_id] = raw_text_lbl
        proc_text_labels[lipd_id] = proc_text_lbl
        action_cat_labels[lipd_id] = action_cat_lbl

        ### Merge the labels with the LIPD dataset.
        if abs(len(poses) - len(lipd[lipd_id]["PCD"])) <= 2:  
            lipd[lipd_id]["amass_poses"] = poses[:len(lipd[lipd_id]["PCD"])]
            lipd[lipd_id]["raw_text"] = raw_text_lbl[:len(lipd[lipd_id]["PCD"])] # not sure how offsync they are
            lipd[lipd_id]["proc_text"] = proc_text_lbl[:len(lipd[lipd_id]["PCD"])] # not sure how offsync they are but thats the best we can do.
            lipd[lipd_id]["action_cat"] = action_cat_lbl[:len(lipd[lipd_id]["PCD"])]
            lipd_valid.append(lipd_id)

    frame_diffs = []
    for lipd_id in lipd_in_amass.keys():
        poses_temp = len(amass_poses[lipd_id])
        poses_lipd = len(lipd[lipd_id]["PCD"])
        frame_diffs.append( [poses_lipd, poses_temp] )
    frame_diffs = np.array(frame_diffs)

    return amass_poses, raw_text_labels, proc_text_labels, action_cat_labels, frame_diffs, lipd, lipd_valid



if __name__ == "__main__":
    # Load Babel

    data_root = "/data"
    d_folder = '/data/BABEL/babel_v1.0_release' # Data folder for babel
    amass_data_root_path = "/data/AMASS"

    l_babel_dense_files = ['train', 'val', 'test']  
    l_babel_extra_files = ['extra_train', 'extra_val']

    # BABEL Dataset 
    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(ospj(d_folder, file+'.json')))
        
    for file in l_babel_extra_files:
        babel[file] = json.load(open(ospj(d_folder, file+'.json'))) 

    
    ### Need to load preprocessed LIPD to merge
    with open('%s/LIPD/LIPD_SEQUENCES_256p.pkl' % data_root, 'rb') as f:
        sequence_datasets = pickle.load(f)

    proc_sequences = {k : sequence_datasets[k] for k in ["ACCAD", "BMLmovi", "CMU", "eTC"]}

    # LIPD-babel-v1
    seqs_with_labels = {}
    for m in ["ACCAD", "BMLmovi", "CMU", "eTC"]:
        if m == "eTC":
            target_fps = 5
        else:
            target_fps = 10
        amass_poses, raw_text_labels, proc_text_labels, action_cat_labels, frame_diffs, lipd_updated, lipd_valid = get_babel_labels_and_amass_poses(sequence_datasets, m, 
                                                                                                                        babel,babel_split="train", 
                                                                                                                        amass_data_root_path=amass_data_root_path,
                                                                                                                        target_fps=target_fps)
        seqs_with_labels[m] = { lid : lipd_updated[lid] for lid in lipd_valid} # only store the ones where we have correct labels.

    ## save
    lipd_babel_train_path = "%s/LIPD/lipd_babel_annotations_GITHUBTEST.pkl" % data_root
    with open(lipd_babel_train_path, "wb") as f:
        pickle.dump(seqs_with_labels, f)

    # LIPD-babel-v2
    seqs_with_labels_val = {}
    for m in ["ACCAD", "BMLmovi", "CMU", "eTC"]:
        if m == "eTC":
            target_fps = 5
        else:
            target_fps = 10
        amass_poses, raw_text_labels, proc_text_labels, action_cat_labels, frame_diffs, lipd_updated, lipd_valid = get_babel_labels_and_amass_poses(sequence_datasets, m, 
                                                                                                                                    babel, babel_split="val", 
                                                                                                                                    amass_data_root_path=amass_data_root_path,
                                                                                                                                    target_fps=target_fps)
        seqs_with_labels_val[m] = { lid : lipd_updated[lid] for lid in lipd_valid} # only store the ones where we have correct labels.

    lipd_babel_val_path = "%s/LIPD/lipd_babel_annotations_VALGITHUBTEST.pkl" % data_root
    with open(lipd_babel_val_path, "wb") as f:
        pickle.dump(seqs_with_labels_val, f)

    ## Could also do this for testset, but testset labels are private, so no use here
    #seqs_with_labels_test = {}
    #for m in ["ACCAD", "BMLmovi", "CMU", "eTC"]:
    #    if m == "eTC":
    #        target_fps = 5
    #    else:
    #        target_fps = 10
    #    amass_poses, raw_text_labels, proc_text_labels, action_cat_labels, frame_diffs, lipd_updated, lipd_valid = amass_preprocessing.get_babel_labels_and_amass_poses(sequence_datasets, m, babel, babel_split="test", amass_data_root_path=amass_data_root_path, target_fps=target_fps)
    #    seqs_with_labels_test[m] = { lid : lipd_updated[lid] for lid in lipd_valid} # only store the ones where we have correct labels.
