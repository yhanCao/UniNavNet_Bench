import os
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import yaml
import cv2
import pickle

radian = 180 / np.pi


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg

def load_config(path, default_path=None):
    """
    Loads config file.
    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
    Returns:
        cfg (dict): config dict.
    """
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def camera_param_to_camera_matrix(camera_param):
    f = camera_param.f
    cx = camera_param.xc
    cy = camera_param.zc
    camera_matrix = np.array([[f, 0.0, cx, 0.0],
                              [0.0, f, cy, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
    return camera_matrix


# @brief: depth preprocessing used by an early test version, unused since v1.
def preprocess_depth(depth, min_d, max_d):
    depth = depth[:, :, 0] * 1

    mask2 = depth > 0.99
    mask1 = depth == 0
    depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
    depth[mask1] = 0.
    depth[mask2] = 0.

    return depth

# @brief: saves the 640 x 480 image

# @brief: saves the inputs of an intermediate scene graph (rgb, depth, objects, pose), where
def save_intermediate_element_for_SGG(obs, pcd_frame, idx,pose,cam_K, path='/home/caoyihan/Code/LLM-SG/outputs', flag_save_rgb=True):
    image_dir_save = os.path.join(path, 'image')
    if not os.path.exists(image_dir_save):
        os.makedirs(image_dir_save)
    pkl_dir_save = os.path.join(path, 'pkl')
    if not os.path.exists(pkl_dir_save):
        os.makedirs(pkl_dir_save)
    save_dict = {}

    _obs = obs.squeeze().permute(1, 2, 0)
    depth = preprocess_depth(_obs[:, :, 3, None], 0.5, 5.0)
    rgb = _obs[:, :, :3]

    #     pcd_data = {
    #         'points': np.asarray(pcd.points),
    #         'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
    #         'normals': np.asarray(pcd.normals) if pcd.has_normals() else None
    #     }
    #         ob['bbox'] = {
    #             'min_bound': np.asarray(ob['bbox'].min_bound),
    #             'max_bound': np.asarray(ob['bbox'].max_bound)
    #         }
    #     elif isinstance(ob['bbox'], o3d.geometry.OrientedBoundingBox):
    #         ob['bbox'] = {
    #             'center': np.asarray(ob['bbox'].center),
    #             'R': np.asarray(ob['bbox'].R),
    #             'extent': np.asarray(ob['bbox'].extent)
    #         }

    #     pcd_data = {
    #         'points': np.asarray(pcd.points),
    #         'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
    #         'normals': np.asarray(pcd.normals) if pcd.has_normals() else None
    #     }

    #         _bg_objects[ob]['bbox'] = {
    #             'min_bound': np.asarray(_bg_objects[ob]['bbox'].min_bound),
    #             'max_bound': np.asarray(_bg_objects[ob]['bbox'].max_bound)
    #         }
    #     elif isinstance(_bg_objects[ob]['bbox'], o3d.geometry.OrientedBoundingBox):
    #         _bg_objects[ob]['bbox']= {
    #             'center': np.asarray(_bg_objects[ob]['bbox'].center),
    #             'R': np.asarray(_bg_objects[ob]['bbox'].R),
    #             'extent': np.asarray(_bg_objects[ob]['bbox'].extent)
    #         }

    #     pcd_data = {
    #         'points': np.asarray(pcd.points),
    #         'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
    #         'normals': np.asarray(pcd.normals) if pcd.has_normals() else None
    #     }

    #         ob['bbox'] = {
    #             'min_bound': np.asarray(ob['bbox'].min_bound),
    #             'max_bound': np.asarray(ob['bbox'].max_bound)
    #         }
    #     elif isinstance(ob['bbox'], o3d.geometry.OrientedBoundingBox):
    #         ob['bbox']= {
    #             'center': np.asarray(ob['bbox'].center),
    #             'R': np.asarray(ob['bbox'].R),
    #             'extent': np.asarray(ob['bbox'].extent)
    #         }

    #     pcd_data = {
    #         'points': np.asarray(pcd.points),
    #         'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
    #         'normals': np.asarray(pcd.normals) if pcd.has_normals() else None
    #     }

    #         ob['bbox'] = {
    #             'min_bound': np.asarray(ob['bbox'].min_bound),
    #             'max_bound': np.asarray(ob['bbox'].max_bound)
    #         }
    #     elif isinstance(ob['bbox'], o3d.geometry.OrientedBoundingBox):
    #         ob['bbox']= {
    #             'center': np.asarray(ob['bbox'].center),
    #             'R': np.asarray(ob['bbox'].R),
    #             'extent': np.asarray(ob['bbox'].extent)
    #         }


    _pcd_frame = {
        'points': pcd_frame.cpu().numpy()
    }
    save_dict['depth'] = depth
    save_dict['rgb'] = rgb
    save_dict['pose'] = pose
    save_dict['cam_K'] = cam_K
    save_dict['pcd_frame'] = _pcd_frame
    save_dict['idx'] = idx
    with open(os.path.join(pkl_dir_save, 'save_dict_{}.pkl'.format(idx)), 'wb') as tf:
        pickle.dump(save_dict, tf)

    if flag_save_rgb:
        save_rgb = rgb.cpu().detach().numpy().astype(np.uint8)
        save_rgb = cv2.cvtColor(save_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_dir_save, '{}.jpg'.format(idx)), save_rgb)

    print('save_intermediate_element_for_SGG')


