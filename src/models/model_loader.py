import argparse
from types import SimpleNamespace
import torch
import clip

# Model Imports
import src.models.encoders as Models
from src.models.motion_clip import Encoder_TRANSFORMER, Decoder_TRANSFORMER


def load_smpl_generator(embed_dim, n_joints, n_feats, device="cuda"):
    model = Decoder_TRANSFORMER("s_generator", n_joints, n_feats, latent_dim=embed_dim).to(device)
    return model

def load_skeleton_generator(embed_dim, n_joints, n_feats, device="cuda"):
    model = Decoder_TRANSFORMER("s_generator", n_joints, n_feats, latent_dim=embed_dim).to(device)
    return model

def load_skeleton_encoder(embed_dim, n_joints, n_feats, device="cuda"):
    #MOTIONCLIP params

    # modeltype, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"
    parameters = {'cuda': True, 
                'device': 0, 
                'modelname': 'motionclip_transformer_rc_rcxyz_vel', 
                'latent_dim': embed_dim, 
                'num_layers': 8, 'activation': 'gelu', 
                'modeltype': 'motionclip'}
    parameters["nfeats"] = n_feats #  only 3d joint coordinates used so its same as NTU
    # LIPD unfort has 24 joints and subset of AMASS is included in there that we could change to 25 joints, but we couldnt change the rest of it to 25.
    parameters["njoints"] = n_joints # smpl 25 joints would be better but dont have pointclouds otherwise.
    smpl_model = Encoder_TRANSFORMER(**parameters).to(device)
    return smpl_model

def load_smpl_encoder(embed_dim, n_joints, n_feats, device="cuda"):
    #MOTIONCLIP params

    # modeltype, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"
    parameters = {'cuda': True, 
                'device': 0, 
                'modelname': 'motionclip_transformer_rc_rcxyz_vel', 
                'latent_dim': embed_dim, 
                'num_layers': 8, 'activation': 'gelu', 
                'modeltype': 'motionclip'}
    parameters["nfeats"] = n_feats #  only 3d joint coordinates used so its same as NTU
    # LIPD unfort has 24 joints and subset of AMASS is included in there that we could change to 25 joints, but we couldnt change the rest of it to 25.
    parameters["njoints"] = n_joints # smpl 25 joints would be better but dont have pointclouds otherwise.
    smpl_model = Encoder_TRANSFORMER(**parameters).to(device)
    return smpl_model

def load_imu_encoder(embed_dim, device="cuda"):
    imu_model = Models.IMUEncoder(input_size=48, hidden_size=embed_dim, num_layers=2, device=device).to(device)
    return imu_model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PST-Transformer Model Training')

    parser.add_argument('--data-path', default='/data/MSR-Action3D/data', type=str, help='dataset')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='PSTTransformer', type=str, help='model')
    # input
    parser.add_argument('--clip-len', default=24, type=int, metavar='N', help='number of frames per clip')
    parser.add_argument('--num-points', default=2048, type=int, metavar='N', help='number of points per frame')
    # P4D
    parser.add_argument('--radius', default=0.3, type=float, help='radius for the ball query')
    parser.add_argument('--nsamples', default=32, type=int, help='number of neighbors for the ball query')
    parser.add_argument('--spatial-stride', default=32, type=int, help='spatial subsampling rate')
    parser.add_argument('--temporal-kernel-size', default=3, type=int, help='temporal kernel size')
    parser.add_argument('--temporal-stride', default=2, type=int, help='temporal stride')
    # transformer
    parser.add_argument('--dim', default=80, type=int, help='transformer dim')
    parser.add_argument('--depth', default=5, type=int, help='transformer depth')
    parser.add_argument('--heads', default=2, type=int, help='transformer head')
    parser.add_argument('--dim-head', default=40, type=int, help='transformer dim for each head')
    parser.add_argument('--mlp-dim', default=160, type=int, help='transformer mlp dim')
    parser.add_argument('--dropout1', default=0.0, type=float, help='transformer dropout')
    # output
    parser.add_argument('--dropout2', default=0.0, type=float, help='classifier dropout')
    # training
    parser.add_argument('-b', '--batch-size', default=14, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
    # output
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', type=str, help='path where to save')
    # resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')

    #args = parser.parse_args("")
    return parser

def load_pst_transformer(embed_dim, device="cuda"):
    #PST Transformer params
    parser = parse_args()
    args = {action.dest: action.default for action in parser._actions if action.default is not argparse.SUPPRESS}
    args = SimpleNamespace(**args)

    Model = getattr(Models, args.model)
    pc_model = Model(radius=args.radius, nsamples=args.nsamples, spatial_stride=args.spatial_stride,
                temporal_kernel_size=args.temporal_kernel_size, temporal_stride=args.temporal_stride,
                dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, dropout1=args.dropout1,
                mlp_dim=embed_dim, num_classes=embed_dim, dropout2=args.dropout2).to(device)

    return pc_model