import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_joint_pcd_ax(ax, joints, pcd, rot=70):  
    ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=3)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    edges = [(0,1), (0,2), (0,3), (1,4),
            (2,5), (3,6), (4,7), (5,8), 
            (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18),
            (17,19), (18,20), (19, 21), (20,22), (21,23)]

    ax.view_init(0, rot)
    for edge in edges:
        ax.plot([joints[edge[0], 0], joints[edge[1], 0]],
                [joints[edge[0], 1], joints[edge[1], 1]],
                [joints[edge[0], 2], joints[edge[1], 2]], c='b')
    plt.axis("off")

def plot_joint_pcd(joints, pcd, rot=70, figsize=(10,10), save_fig=False, path=""):  
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=3)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    edges = [(0,1), (0,2), (0,3), (1,4),
            (2,5), (3,6), (4,7), (5,8), 
            (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18),
            (17,19), (18,20), (19, 21), (20,22), (21,23)]

    ax.view_init(0, rot)
    for edge in edges:
        ax.plot([joints[edge[0], 0], joints[edge[1], 0]],
                [joints[edge[0], 1], joints[edge[1], 1]],
                [joints[edge[0], 2], joints[edge[1], 2]], c='b')
    plt.axis("off")

    if save_fig:
        plt.savefig(path)
        plt.close()


def plot_joints(joints_a, joints_b, rot=70, figsize=(10,5), edges_b=None):
    # Plot the 3D joints
    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=figsize)
    # Plot each joint as a point
    #ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    ax[0].scatter(joints_a[:, 0], joints_a[:, 1], joints_a[:, 2], c='r', marker='o')
    ax[0].set_xlim([-1,1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_zlim([-1, 1])

    # based on https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    edges = [(0,1), (0,2), (0,3), (1,4),
            (2,5), (3,6), (4,7), (5,8), 
            (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18),
            (17,19), (18,20), (19, 21), (20,22), (21,23)]  ### without (20,22) and (21,23) we have HumanML3D 22 joints

    ax[0].view_init(0, rot)

    for edge in edges:
        ax[0].plot([joints_a[edge[0], 0], joints_a[edge[1], 0]],
                [joints_a[edge[0], 1], joints_a[edge[1], 1]],
                [joints_a[edge[0], 2], joints_a[edge[1], 2]], c='b')

    
    ax[1].scatter(joints_b[:, 0], joints_b[:, 1], joints_b[:, 2], c='r', marker='o')
    ax[1].set_xlim([-1,1])
    ax[1].set_ylim([-1, 1])
    ax[1].set_zlim([-1, 1])

    # based on https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    if edges_b:
        edges = edges_b
    else:
        edges = [(0,1), (0,2), (0,3), (1,4),
                (2,5), (3,6), (4,7), (5,8), 
                (6,9), (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17), (16,18),
                (17,19), (18,20), (19, 21), (20,22), (21,23)]  ### without (20,22) and (21,23) we have HumanML3D 22 joints

    ax[1].view_init(0, rot)

    for edge in edges:
        ax[1].plot([joints_b[edge[0], 0], joints_b[edge[1], 0]],
                [joints_b[edge[0], 1], joints_b[edge[1], 1]],
                [joints_b[edge[0], 2], joints_b[edge[1], 2]], c='b')


def example():
        from src import amass_preprocessing
        from human_body_prior.body_model.body_model import BodyModel
        from src import visualization
        comp_device = "cpu"
        SMPLH_AMASS_MODEL_PATH = "../MotionCLIP/models/smplh/neutral/model.npz"
        NUM_BETAS = 10
        body_models = {
                'neutral': BodyModel(SMPLH_AMASS_MODEL_PATH, num_betas=NUM_BETAS).to(comp_device),
        }
        ### Make sure to use same joints. TEST
        joints_to_use = np.array([0,1,2,3,4,5,6,7,8,9,10, 11, 12,13,14, 15, 16,17,18,19,20,21,22,37])
        bdata = np.load("/data/AMASS/BMLmovi/Subject_40_F_MoSh/Subject_40_F_19_poses.npz")
        amass_joints = amass_preprocessing.amass_poses_and_trans(bdata, body_model=body_models['neutral'], joints_to_use=joints_to_use)
        poses = bdata['poses'][:, joints_to_use]
        time = 50
        visualization.plot_joints(amass_joints[::12][time], lipd_joints[time], figsize=(20,10))