from pathlib import Path
import warnings
from rich.console import Console

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    warnings.warn("omegaconf not available. Install with: pip install omegaconf")


# Some of the code was built off open-source code in https://github.com/world-model-eval/world-model-eval


# console
console = Console()


def _load_config(config_path: str | Path):
    """Load configuration from YAML file using OmegaConf."""
    if not OMEGACONF_AVAILABLE:
        raise ImportError("omegaconf is required. Install with: pip install omegaconf")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    console.log("Loaded config from %s", config_path)
    return cfg

def find_target_path_by_name(root_directory, target_name_exp):
    """
    Finds all subfolders with a given name within a root directory.

    Args:
        root_directory (str): The path to the directory to start searching from.
        target_name_exp (str): The name of the subfolders to find or expression.

    Returns:
        list: A list of full paths to the found subfolders.
    """
    # path to target subfolders
    target_paths = [
        d_path for d_path in Path(root_directory).rglob(target_name_exp)
    ]
    
    return target_paths