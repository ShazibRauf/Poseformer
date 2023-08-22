# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.model_poseformer import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator, ChunkedGenerator_stage3
from time import time
from common.utils import *

import FrEIA.framework as Ff
import FrEIA.modules as Fm


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parse_args()
print('Training Curves:', args.export_training_curves)

num_bases = 26
bone_relations_mean = torch.Tensor([0.5181, 1.7371, 1.7229, 0.5181, 1.7371, 1.7229, 0.9209, 0.9879,
                                0.4481, 0.4450, 0.5746, 1.0812, 0.9652, 0.5746, 1.0812, 0.9652]).cuda()
depth_offset = 10.0

inn_2d = Ff.SequenceINN(num_bases)
for k in range(8):
    inn_2d.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)


inn_2d.load_state_dict(torch.load('models/model_inn_h36m_17j_pretrain_inn_gt_pca_bases_%d_headnorm.pt' % num_bases))
# freeze all weights in INN
for param in inn_2d.parameters():
    param.requires_grad = False

inn_2d = inn_2d.cuda()



try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)


receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = 100000
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

#########################################PoseTransformer

model_pos_train = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)

################ load weight ########################
# posetrans_checkpoint = torch.load('./checkpoint/pretrained_posetrans.bin', map_location=lambda storage, loc: storage)
# posetrans_checkpoint = posetrans_checkpoint["model_pos"]
# model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)


test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


###################

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []  
    losses_3d_train_step1 = []  
    losses_3d_train_step2 = [] 
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_3d_train_eval1 = []
    losses_3d_valid1 = []

    ls_likeli = []
    ls_L3d = []
    ls_rep_rot = []
    ls_re_rot_3d = []
    ls_bl_prior = []


    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    train_generator_stage2 = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, stage=2)
    
    
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    
    pca = train_generator.pca
    

    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))


    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']


    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    ######################################################################################################################
    ######################################################################################################################
    ###############################################   STAGE:1    #########################################################
    ######################################################################################################################
    ######################################################################################################################
    # stage1 - training
    while epoch < args.epochs:
        print('################### EPOCH: ',epoch,'######################')
        print('stage 1 activated.....')
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        model_pos_train.train()

        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
            inputs_3d[:, :, 0] = 0


            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos, _ = model_pos_train(inputs_2d)

            pred = predicted_3d_pos.squeeze(1)[:, :, -1]
            pred[:, 0] = 0.0

            depth = pred + depth_offset
            depth[depth < 1.0] = 1.0
            pred_3d = torch.cat(((inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17)

            proj_2d = perspective_projection(pred_3d.reshape(-1, 51))
            loss_2d_pos = (proj_2d - inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1).reshape(-1, 2*num_joints)).abs().sum(dim=1).mean()

            del inputs_2d
            torch.cuda.empty_cache()


            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_2d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            #Loss
            loss_total = loss_2d_pos
            loss_total.backward()

            optimizer.step()
            del inputs_3d, loss_2d_pos, predicted_3d_pos
            torch.cuda.empty_cache()
            #break #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        losses_3d_train_step1.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()



        ######################################################################################################################
        ######################################################################################################################
        ###############################################   STAGE:2    #########################################################
        ######################################################################################################################
        ######################################################################################################################

        poses_3d_stage2 = train_generator_stage2.poses_3d
        print('stage 2 activated.....')

        # stage2 - evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_stage2 = 0
            N = 0
            results=[]
            pairs = []
            elevation = {}
            for _, batch_3d, batch_2d, chunks in train_generator_stage2.next_epoch():
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                inputs_3d[:, :, 0] = 0

                # Predict 3D poses
                predicted_3d_pos, elev = model_pos(inputs_2d)
                results = predicted_3d_pos.squeeze(1).cpu().numpy()

                #Saving Predictions and Elevation
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    if (seq_i  not in elevation.keys()):
                        elevation[seq_i] = {} 

                    elevation[seq_i].update({start_3d: elev[i].cpu().numpy()})
                    poses_3d_stage2[seq_i][start_3d] = results[i]

                if (N == 0):
                    pairs = chunks
                else:
                    pairs = np.concatenate((pairs, chunks), axis=0)

                #if (N != 0): #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                #    break

                del inputs_2d
                torch.cuda.empty_cache()


                loss_3d_pos = n_mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_stage2 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                del inputs_3d, loss_3d_pos, predicted_3d_pos, elev
                torch.cuda.empty_cache()
            
            losses_3d_train_step2.append(epoch_loss_3d_stage2 / N)



        ######################################################################################################################
        ######################################################################################################################
        ###############################################   STAGE:3    #########################################################
        ######################################################################################################################
        ######################################################################################################################


        train_generator_stage3 = ChunkedGenerator_stage3(args.batch_size//args.stride, cameras_train, poses_3d_stage2, poses_train_2d, args.stride,
                                    pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right, pairs=pairs, elevation=elevation)


        print('stage 3 activated.....')
        N = 0
        epoch_loss = 0
        #model_pos_train.train()
        epoch_likeli = 0
        epoch_L3d = 0
        epoch_rep_rot = 0
        epoch_re_rot_3d = 0
        epoch_bl_prior = 0


        for _, batch_3d, batch_2d, batch_elevation in train_generator_stage3.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            props = torch.from_numpy(batch_elevation.astype('float32'))
            pred = inputs_3d.reshape(inputs_3d.shape[0]*inputs_3d.shape[1], 17, 3).permute(0, 2, 1)

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                props = props.cuda()
                pred = pred.cuda()

            optimizer.zero_grad()


            x_ang_comp = torch.ones((pred.shape[0], 1)).cuda() * props
            y_ang_comp = torch.zeros((pred.shape[0], 1)).cuda()
            z_ang_comp = torch.zeros((pred.shape[0], 1)).cuda()

            euler_angles_comp = torch.cat((x_ang_comp, y_ang_comp, z_ang_comp), dim=1)
            R_comp = euler_angles_to_matrix(euler_angles_comp, 'XYZ')


            # sample from learned distribution
            elevation = torch.cat((props.mean().reshape(1), props.std().reshape(1)))

            x_ang = (-elevation[0]) + elevation[1] * torch.normal(torch.zeros((pred.shape[0], 1)).cuda(), torch.ones((pred.shape[0], 1)).cuda())
            y_ang = (torch.rand((pred.shape[0], 1)).cuda() - 0.5) * 2.0 * np.pi
            z_ang = torch.zeros((pred.shape[0], 1)).cuda()

            Rx = euler_angles_to_matrix(torch.cat((x_ang, z_ang, z_ang), dim=1), 'XYZ')
            Ry = euler_angles_to_matrix(torch.cat((z_ang, y_ang, z_ang), dim=1), 'XYZ')

            R = Rx @ (Ry @ R_comp)


            depth = pred[:, -1, :] + depth_offset
            depth[depth < 1.0] = 1.0
            pred_3d = torch.cat(((inputs_2d.reshape(inputs_3d.shape[0]*inputs_3d.shape[1], 17, 2).permute(0, 2, 1) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17)

            #Root Relative
            pred_3d = pred_3d.reshape(-1, 3, num_joints) - pred_3d.reshape(-1, 3, num_joints)[:, :, [0]]
            rot_poses = (R.matmul(pred_3d)).reshape(-1, 51)

            ## lift from augmented camera and normalize
            global_pose = torch.cat((rot_poses[:, 0:34], rot_poses[:, 34:51] + depth_offset), dim=1)
            rot_2d = perspective_projection(global_pose)
            norm_poses = rot_2d
            
            norm_poses_mean = norm_poses[:, 0:34] - torch.Tensor(pca.mean_.reshape(1, 34)).cuda()
            latent = norm_poses_mean @ torch.Tensor(pca.components_.T).cuda()
            
            z, log_jac_det = inn_2d(latent[:, 0:num_bases])
            likelis = 0.5 * torch.sum(z ** 2, 1) - log_jac_det

            losses_likeli = likelis.mean()
            epoch_likeli += inputs_3d.shape[0] * losses_likeli.cpu()
            #print('losses_likeli',losses_likeli)

            ## reprojection error
            norm_poses = norm_poses.reshape(-1, 2, num_joints).permute(0, 2, 1).reshape(inputs_3d.shape[0], inputs_3d.shape[1], num_joints, 2)
            pred_rot, _ = model_pos_train(norm_poses)
            pred_rot = pred_rot.squeeze(1)[:, :, -1]
            pred_rot[:, 0] = 0.0


            pred_rot_depth = pred_rot + depth_offset
            pred_rot_depth[pred_rot_depth < 1.0] = 1.0
            pred_3d_rot = torch.cat(((norm_poses[:, int(receptive_field/2), :, :].permute(0, 2, 1) * pred_rot_depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), pred_rot_depth), dim=1)


            pred_3d_rot = pred_3d_rot.reshape(-1, 3, num_joints) - pred_3d_rot.reshape(-1, 3, num_joints)[:, :, [0]]
            losses_L3d = (rot_poses.reshape(inputs_3d.shape[0], inputs_3d.shape[1], 51)[:, int(receptive_field/2), :] - pred_3d_rot.reshape(-1, 51)).norm(dim=1).mean()
            epoch_L3d += inputs_3d.shape[0] * losses_L3d.cpu().detach().numpy()
            #print('losses_L3d',losses_L3d)

            
            R = R.reshape(inputs_3d.shape[0], inputs_3d.shape[1], 3, 3).mean(axis=1)
            re_rot_3d = (R.permute(0, 2, 1) @ pred_3d_rot).reshape(-1, 51)
            pred_rot_global_pose = torch.cat((re_rot_3d[:, 0:34], re_rot_3d[:, 34:51] + depth_offset), dim=1)
            #pred_rot_global_pose = re_rot_3d
            re_rot_2d = perspective_projection(pred_rot_global_pose)
            norm_re_rot_2d = re_rot_2d


            losses_rep_rot = (norm_re_rot_2d - inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1).reshape(-1, 2*num_joints)).abs().sum(dim=1).mean()
            epoch_rep_rot += inputs_3d.shape[0] * losses_rep_rot.cpu().detach().numpy()
            #print('losses_rep_rot', losses_rep_rot)

            

            pred_3d = pred_3d.reshape(inputs_3d.shape[0], inputs_3d.shape[1], 3, 17)[:, int(receptive_field/2), :, :]       
            # pairwise deformation loss
            num_pairs = int(np.floor(pred_3d.shape[0] / 2))
            pose_pairs = pred_3d[0:(2 * num_pairs)].reshape(2 * num_pairs, 51).reshape(-1, 2, 51)
            pose_pairs_re_rot_3d = re_rot_3d[0:(2*num_pairs)].reshape(-1, 2, 51)
            losses_re_rot_3d = ((pose_pairs[:, 0] - pose_pairs[:, 1]) - (pose_pairs_re_rot_3d[:, 0] - pose_pairs_re_rot_3d[:, 1])).norm(dim=1).mean()
            epoch_re_rot_3d += inputs_3d.shape[0] * losses_re_rot_3d.cpu().detach().numpy()
            #print('losses_re_rot_3d', losses_re_rot_3d)
            

            ## bone lengths prior
            bl = get_bone_lengths_all(pred_3d.reshape(-1, 51))
            rel_bl = bl / bl.mean(dim=1, keepdim=True)
            losses_bl_prior = (bone_relations_mean - rel_bl).square().sum(dim=1).mean()
            epoch_bl_prior += inputs_3d.shape[0] * losses_bl_prior.cpu().detach().numpy()
            #print('losses_bl_prior',losses_bl_prior)

            weight_bl = 80.0
            weight_2d = 1.0
            weight_3d = 1.0
            weight_velocity = 1.0

            losses_loss = losses_likeli + \
                        weight_2d*losses_rep_rot + \
                        weight_3d * losses_L3d + \
                        weight_velocity*losses_re_rot_3d

            losses_loss = losses_loss + weight_bl*losses_bl_prior

            epoch_loss += inputs_3d.shape[0] * losses_loss.item()
            N += inputs_3d.shape[0]

            losses_loss.backward()
            optimizer.step()

            del inputs_2d, inputs_3d, pred, props
            torch.cuda.empty_cache()
            #break

        losses_3d_train.append(epoch_loss / N)
        ls_likeli.append(epoch_likeli / N)
        ls_L3d.append(epoch_L3d / N)
        ls_rep_rot.append(epoch_rep_rot / N)
        ls_re_rot_3d.append(epoch_re_rot_3d / N)
        ls_bl_prior.append(epoch_bl_prior / N)

        ######################################################################################################################
        ######################################################################################################################
        ###############################################   EVALUATION    ######################################################
        ######################################################################################################################
        ######################################################################################################################


        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_3d_valid1 = 0
            N = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_3d = inputs_3d.cuda()
                    inputs_3d[:, :, 0] = 0

                    #Predictions
                    predicted_3d_pos, _ = model_pos(inputs_2d)
                    
                    pred = predicted_3d_pos.squeeze(1)[:, :, -1]
                    pred[:, 0] = 0.0
                    depth = pred + depth_offset
                    predicted_3d_pos = torch.cat(((inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17).permute(0, 2, 1)

                    
                    loss_3d_pos1 = n_mpjpe_updated(predicted_3d_pos.permute(0, 2, 1).reshape(-1, 51), inputs_3d.squeeze(1).permute(0, 2, 1).reshape(-1, 51))
                    epoch_loss_3d_valid1 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos1.item() #<<<

                    proj_2d = perspective_projection(predicted_3d_pos.permute(0, 2, 1).reshape(-1, 51))
                    loss_3d_pos = (proj_2d - inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1).reshape(-1, 2*num_joints)).abs().sum(dim=1).mean()
                    
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    del inputs_3d, loss_3d_pos, predicted_3d_pos, inputs_2d
                    torch.cuda.empty_cache()
                    #break

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                losses_3d_valid1.append(epoch_loss_3d_valid1 / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_3d_train_eval1 = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()

                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos, _ = model_pos(inputs_2d)

                    pred = predicted_3d_pos.squeeze(1)[:, :, -1]
                    pred[:, 0] = 0.0
                    depth = pred + depth_offset
                    predicted_3d_pos = torch.cat(((inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17).permute(0, 2, 1).reshape(-1, 17, 3)

                    loss_3d_pos1 = n_mpjpe_updated(predicted_3d_pos.permute(0, 2, 1).reshape(-1, 51), inputs_3d.squeeze(1).permute(0, 2, 1).reshape(-1, 51))#<<
                    epoch_loss_3d_train_eval1 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos1.item() #<<<


                    proj_2d = perspective_projection(predicted_3d_pos.permute(0, 2, 1).reshape(-1, 51))
                    loss_3d_pos = (proj_2d - inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1).reshape(-1, 2*num_joints)).abs().sum(dim=1).mean()
                    

                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    del inputs_3d, loss_3d_pos, predicted_3d_pos, inputs_2d
                    torch.cuda.empty_cache()
                    #break

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
                losses_3d_train_eval1.append(epoch_loss_3d_train_eval1 / N)


        elapsed = (time() - start_time) / 60    
               

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train (Proj) %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:

            print('[%d] time %.2f lr  %f 3d_train (Proj-step1) %f 3d_val (Step2) %f 3d_train (Step3) %f 3d_eval(MAE) %f 3d_valid(MAE) %f 3d_eval(MPJPE) %f 3d_valid(MPJPE) %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train_step1[-1] * 1000,
                losses_3d_train_step2[-1] * 1000,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000,
                losses_3d_train_eval1[-1] * 1000,
                losses_3d_valid1[-1] * 1000))

            print('[%d] ls_likeli %f ls_L3d %f ls_rep_rot %f ls_re_rot_3d %f ls_bl_prior %f' % (
                epoch + 1,
                ls_likeli[-1] * 1000,
                ls_L3d[-1] * 1000,
                ls_rep_rot[-1] * 1000,
                ls_re_rot_3d[-1] * 1000,
                ls_bl_prior[-1] * 1000))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch >= 0:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train_eval1)) + 1
            plt.plot(epoch_x, losses_3d_train_eval1[0:], color='C0')
            plt.plot(epoch_x, losses_3d_valid1[0:], color='C1')
            plt.legend(['3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'validation_loss_3d_mpjpe.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_valid1)) + 1
            plt.plot(epoch_x, losses_3d_valid1[0:], color='C1')
            plt.legend(['3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'valset_loss_3d_mpjpe.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train_eval1)) + 1
            plt.plot(epoch_x, losses_3d_train_eval1[0:], color='C0')
            plt.legend(['3d train (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'trainingset_eval_loss_3d_mpjpe.png'))
            plt.close('all')


            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train_eval)) + 1
            plt.plot(epoch_x, losses_3d_train_eval[0:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[0:], color='C1')
            plt.legend(['3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'validation_loss_3d_mae.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train_step1)) + 1
            plt.plot(epoch_x, losses_3d_train_step1[0:], color='C1')
            plt.legend(['Training Loss (step1)'])
            plt.ylabel('Combine Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'Training_loss_3d_step1.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train_step2)) + 1
            plt.plot(epoch_x, losses_3d_train_step2[0:], color='C1')
            plt.legend(['Training Loss (step2)'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'Training_loss_3d_step2.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[0:], color='C0')
            plt.legend(['Training Loss (step3)'])
            plt.ylabel('Combine Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'Training_loss_3d_step3.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(ls_likeli)) + 1
            plt.plot(epoch_x, ls_likeli[0:], color='C1')
            plt.legend(['Training Loss likeli'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_likeli.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(ls_L3d)) + 1
            plt.plot(epoch_x, ls_L3d[0:], color='C1')
            plt.legend(['Training Loss L3d'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_L3d.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(ls_rep_rot)) + 1
            plt.plot(epoch_x, ls_rep_rot[0:], color='C1')
            plt.legend(['Training Loss rep_rot'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_rep_rot.png'))
            plt.close('all')


            plt.figure()
            epoch_x = np.arange(0, len(ls_re_rot_3d)) + 1
            plt.plot(epoch_x, ls_re_rot_3d[0:], color='C1')
            plt.legend(['Training Loss re_rot_3d'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_re_rot_3d.png'))
            plt.close('all')

            plt.figure()
            epoch_x = np.arange(0, len(ls_bl_prior)) + 1
            plt.plot(epoch_x, ls_bl_prior[0:], color='C1')
            plt.legend(['Training Loss bl_prior'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.xlim((0, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_bl_prior.png'))
            plt.close('all')


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_loss_3d_valid1 = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
            # model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))


            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0

            predicted_3d_pos, _ = model_pos(inputs_2d)
            predicted_3d_pos_flip, _ = model_pos(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            pred = predicted_3d_pos.squeeze(1)[:, :, -1]
            pred[:, 0] = 0.0
            depth = pred + depth_offset
            predicted_3d_pos = torch.cat(((inputs_2d[:, int(receptive_field/2), :, :].permute(0, 2, 1) * depth.reshape(-1, 1, 17).repeat(1, 2, 1)).reshape(-1, 34), depth), dim=1).reshape(-1, 3, 17).permute(0, 2, 1)

           
            loss_3d_pos1 = n_mpjpe_updated(predicted_3d_pos.permute(0, 2, 1).reshape(-1, 51), inputs_3d.squeeze(1).permute(0, 2, 1).reshape(-1, 51)) #<<<<
            epoch_loss_3d_valid1 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos1.item() #<<<

            print(inputs_3d.squeeze(1)[0].detach().cpu().numpy().shape)
            print(predicted_3d_pos[0].detach().cpu().numpy().shape)
            visualize_body_pose(inputs_3d.squeeze(1)[0].detach().cpu().numpy(), predicted_3d_pos[0].detach().cpu().numpy())

            #predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
            #                              keepdim=True)

            del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev

if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))


    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        print("-->",actions)
        for subject, action in actions:
            print('####',subject)
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')