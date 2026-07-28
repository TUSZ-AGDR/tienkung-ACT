from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Parlor scene asset."""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"
PARLOR_SCENE_USD_PATH = str(SCENES_ROOT / "parlor" / "scene_v2.usd")

PARLOR_SCENE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PARLOR_SCENE_USD_PATH,
    ),
)
