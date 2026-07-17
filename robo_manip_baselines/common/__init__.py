from importlib import import_module

from .base.PhaseBase import PhaseBase, ReachPhaseBase, GraspPhaseBase, HSRGraspPhaseBase
from .base.DatasetBase import DatasetBase
from .base.RolloutBase import RolloutBase

from .data.DataKey import DataKey
from .data.CachedDataset import CachedDataset
from .data.EnvDataMixin import EnvDataMixin

from .manager.PhaseManager import PhaseManager
from .manager.MotionManager import MotionManager

from .body.BodyManagerBase import BodyConfigBase, BodyManagerBase
from .body.ArmManager import ArmConfig, ArmManager
from .body.MobileOmniManager import MobileOmniConfig, MobileOmniManager

from .utils.MathUtils import (
    set_random_seed,
    get_pose_from_rot_pos,
    get_rot_pos_from_pose,
    get_pose_from_se3,
    get_se3_from_pose,
    get_rel_pose_from_se3,
    get_se3_from_rel_pose,
)
from .utils.VisionUtils import (
    crop_and_resize,
    convert_depth_image_to_color_image,
    convert_depth_image_to_point_cloud,
)
from .utils.DataUtils import (
    normalize_data,
    denormalize_data,
    get_skipped_data_seq,
    get_skipped_single_data,
)
from .utils.EnvUtils import get_env_names
from .utils.MiscUtils import remove_prefix, remove_suffix


_LAZY_EXPORTS = {
    "TrainBase": ".base.TrainBase",
    "RmbData": ".data.RmbData",
    "DataManager": ".manager.DataManager",
    "DataManagerVec": ".manager.DataManagerVec",
}


def __getattr__(name):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
