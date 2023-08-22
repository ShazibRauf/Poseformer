# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model


#==============================================================================================================================
import functools
import torch.nn as nn

def perspective_projection(pose_3d):

    p2d = pose_3d[:, 0:34].reshape(-1, 2, 17)
    p2d = p2d / pose_3d[:, 34:51].reshape(-1, 1, 17)

    return p2d.reshape(-1, 34)




def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 1024), nn.ReLU(),
                         nn.Linear(1024, dims_out))


def get_bone_lengths_all(poses):
    
    bone_map = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
            [12, 13], [8, 14], [14, 15], [15, 16]]
    

    poses = poses.reshape((-1, 3, 17))

    ext_bones = poses[:, :, bone_map]

    bones = ext_bones[:, :, :, 0] - ext_bones[:, :, :, 1]

    bone_lengths = torch.norm(bones, p=2, dim=1)

    return bone_lengths


import matplotlib.pyplot as plt

def visualize_body_pose(ground_truth, predicted, img="", ground_truth_2D=[], error=0.0):
    # Prepare the figure and axes
    fig = plt.figure(figsize=(2,4))
    plt.gcf().text(0.44, 0.81, 'MPJPE_Scaled: '+str(error), fontsize=14, color='red', weight='bold')
    figManager = plt.get_current_fig_manager()
    figManager.resize(1000,1000)

    # Plot ground truth annotations
    gt_x = ground_truth[:, 0]
    gt_y = ground_truth[:, 1]
    gt_z = ground_truth[:, 2]
    ax = fig.add_subplot(1,4,1, projection='3d')
    ax.scatter(gt_x, gt_y, gt_z, color='blue', label='Ground Truth')

    max_range = None

    if not max_range:
        # Correct aspect ratio (https://stackoverflow.com/a/21765085).
        max_range = (
            np.array(
                [
                    gt_x.max() - gt_x.min(),
                    gt_y.max() - gt_y.min(),
                    gt_z.max() - gt_z.min(),
                ]
            ).max()
            / 2.0
        )
    else:
        max_range /= 2
    mid_x = (gt_x.max() + gt_x.min()) * 0.5
    mid_y = (gt_y.max() + gt_y.min()) * 0.5
    mid_z = (gt_z.max() + gt_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    gt_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]  # Define the connections between keypoints
    itr = 0
    for connection in gt_connections:
        x = [gt_x[connection[0]], gt_x[connection[1]]]
        y = [gt_y[connection[0]], gt_y[connection[1]]]
        z = [gt_z[connection[0]], gt_z[connection[1]]]
        if (itr <= 2 or itr > 12 and itr <= 16):
            ax.plot(x, y, z, color='red')
        elif (itr > 2 and itr <= 5 or itr > 9 and itr <= 12):
            ax.plot(x, y, z, color='lime')
        elif (itr > 5 and itr <= 9):
            ax.plot(x, y, z, color='cyan')
            
        itr = itr + 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ground Truth Annotations')

    
    # Plot predicted annotations
    pred_x = predicted[:, 0]
    pred_y = predicted[:, 1]
    pred_z = predicted[:, 2]
    ax1 = fig.add_subplot(1,4,2, projection='3d')
    ax1.scatter(pred_x, pred_y, pred_z, color='blue', label='Predicted')


    max_range = None

    if not max_range:
        # Correct aspect ratio (https://stackoverflow.com/a/21765085).
        max_range = (
            np.array(
                [
                    pred_x.max() - pred_x.min(),
                    pred_y.max() - pred_y.min(),
                    pred_z.max() - pred_z.min(),
                ]
            ).max()
            / 2.0
        )
    else:
        max_range /= 2
    mid_x = (pred_x.max() + pred_x.min()) * 0.5
    mid_y = (pred_y.max() + pred_y.min()) * 0.5
    mid_z = (pred_z.max() + pred_z.min()) * 0.5

    print('=======', mid_x)
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # Connect predicted keypoints with lines
    pred_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]  # Define the connections between keypoints
    itr = 0
    for connection in pred_connections:
        x = [pred_x[connection[0]], pred_x[connection[1]]]
        y = [pred_y[connection[0]], pred_y[connection[1]]]
        z = [pred_z[connection[0]], pred_z[connection[1]]]
        if (itr <= 2 or itr > 12 and itr <= 16):
            ax1.plot(x, y, z, color='red')
        elif (itr > 2 and itr <= 5 or itr > 9 and itr <= 12):
            ax1.plot(x, y, z, color='lime')
        elif (itr > 5 and itr <= 9):
            ax1.plot(x, y, z, color='cyan')
        itr = itr + 1

    # Set plot limits and labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Predicted Annotations')


    ax2 = fig.add_subplot(1,4,3, projection='3d')
    ax2.scatter(gt_x, gt_y, gt_z, color='blue', label='Ground Truth')

    
    # 0 is the root node
    scale = predicted[0, :] - ground_truth[0, :]

    pred_x = predicted[:, 0] - scale[0]
    pred_y = predicted[:, 1] - scale[1]
    pred_z = predicted[:, 2] - scale[2]
    

    ax2.scatter(pred_x, pred_y, pred_z, color='red', label='Predicted')

    max_range = None

    if not max_range:
        # Correct aspect ratio (https://stackoverflow.com/a/21765085).
        max_range = (
            np.array(
                [
                    pred_x.max() - pred_x.min(),
                    pred_y.max() - pred_y.min(),
                    pred_z.max() - pred_z.min(),
                ]
            ).max()
            / 2.0
        )
    else:
        max_range /= 2
    mid_x = (pred_x.max() + pred_x.min()) * 0.5
    mid_y = (pred_y.max() + pred_y.min()) * 0.5
    mid_z = (pred_z.max() + pred_z.min()) * 0.5

    print('=======', mid_x)
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    itr = 0
    for connection in pred_connections:
        x = [pred_x[connection[0]], pred_x[connection[1]]]
        y = [pred_y[connection[0]], pred_y[connection[1]]]
        z = [pred_z[connection[0]], pred_z[connection[1]]]
        ax2.plot(x, y, z, color='red')

        x = [gt_x[connection[0]], gt_x[connection[1]]]
        y = [gt_y[connection[0]], gt_y[connection[1]]]
        z = [gt_z[connection[0]], gt_z[connection[1]]]
        ax2.plot(x, y, z, color='blue')
        

        itr = itr + 1

     # Set plot limits and labels
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Combined Annotations')

    ax2.legend()

    # Show the plot
    plt.show()