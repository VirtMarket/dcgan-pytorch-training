import os
import yaml
import joblib
from box import Box
from pathlib import Path


def read_config_simple(filename: str) -> Box:
    """Read any yaml file as a Box object"""

    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        with open(filename, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        return Box(config_dict)
    else:
        raise FileNotFoundError(filename)
    

def hasher(x):
    return int(joblib.hash(x), 16)


Box.__hash__ = hasher

def read_config(path: str = "config", *paths):
    """
    get config as dictionary with attribute access to values which can also be hashed
    Args:
        path (str): paths to configuration files or directories with configuration files
        paths (List[str]): additional paths
    Returns:
        box.Box: config as dictionary with attribute access to values
    Examples:
        .. code-block:: python
            from leapfrog.etl import read_config
            # read the default config from the "config" directory and update it with the user config "my_config.yaml"
            config = read_config("config", "my_config.yaml")
    """
    # expand paths to yaml files
    file_paths = []
    for path in [path] + list(paths):
        if path.endswith(".yaml") or path.endswith(".yml"):
            file_paths.append(path)
        else:
            file_paths.extend(list(Path(path).glob("*.yml")))
            file_paths.extend(list(Path(path).glob("*.yaml")))

    # read files
    config = Box()
    for path in file_paths:
        config.merge_update(Box.from_yaml(filename=path))
    return config