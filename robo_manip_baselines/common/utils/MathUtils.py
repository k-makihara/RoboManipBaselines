import random

import numpy as np
import pinocchio as pin
import torch
from typing import List

def set_random_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pose_from_rot_pos(rot, pos):
    """Get pose (tx, ty, tz, qw, qx, qy, qz) from rotation (3D square matrix) and position (3D vector)."""
    return np.concatenate([pos, pin.Quaternion(rot).coeffs()[[3, 0, 1, 2]]])


def get_rot_pos_from_pose(pose):
    """Get rotation (3D square matrix) and position (3D vector) from pose (tx, ty, tz, qw, qx, qy, qz)."""
    return pin.Quaternion(*pose[3:7]).toRotationMatrix(), pose[0:3].copy()


def get_pose_from_se3(se3):
    """Get pose (tx, ty, tz, qw, qx, qy, qz) from pinocchio SE3."""
    return np.concatenate(
        [se3.translation, pin.Quaternion(se3.rotation).coeffs()[[3, 0, 1, 2]]]
    )


def get_se3_from_pose(pose):
    """Get pinocchio SE3 from pose (tx, ty, tz, qw, qx, qy, qz)."""
    return pin.SE3(pin.Quaternion(*pose[3:7]), pose[0:3])


def get_rel_pose_from_se3(se3):
    """Get relative pose (tx, ty, tz, roll, pitch, yaw) from pinocchio SE3."""
    return np.concatenate([se3.translation, pin.rpy.matrixToRpy(se3.rotation)])


def get_se3_from_rel_pose(rel_pose):
    """Get pinocchio SE3 from relative pose (tx, ty, tz, roll, pitch, yaw)."""
    return pin.SE3(pin.rpy.rpyToMatrix(rel_pose[3:6]), rel_pose[0:3])

def rel_pose_eef_to_se3(prev_se3: pin.SE3, rel_pose: List[float]) -> pin.SE3:
    """
    EEF ローカル座標系で表現された相対姿勢 rel_pose を、
    ベース座標系での SE3 コマンドに変換して返す。
    prev_se3 : pin.SE3   
    rel_pose : 長さ6のリスト [x, y, z, roll, pitch, yaw]
    """
    t_rel = np.array(rel_pose[0:3], dtype=float)
    R_rel = pin.rpy.rpyToMatrix(*rel_pose[3:6])
    T_rel = pin.SE3(R_rel, t_rel)

    return prev_se3 * T_rel
