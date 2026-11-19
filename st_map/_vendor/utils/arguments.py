import yaml
import logging
logger = logging.getLogger(__name__)
def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def load_opt_command(runtime_cfg=None):
    from st_map.config import resource_path
    runtime_cfg = runtime_cfg or {}
    opt = load_opt_from_config_files([runtime_cfg.get(
        'openseed_config', resource_path('configs/openseed/openseed_swinl_lang_decouple.yaml'))])
    opt['WEIGHT'] = runtime_cfg.get('openseed_checkpoint', '')
    opt['MODEL']['BACKBONE']['LOAD_PRETRAINED'] = False
    return opt



