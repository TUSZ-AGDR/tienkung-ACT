import gymnasium as gym

gym.register(
    id="LeIsaac-TiangongPro-Parlor-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiangong_parlor_env_cfg:TiangongParlorEnvCfg",
    },
)

