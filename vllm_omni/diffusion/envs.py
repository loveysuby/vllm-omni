# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/envs.py
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: int | None = None
    CUDA_HOME: str | None = None
    LOCAL_RANK: int = 0
    VLLM_OMNI_SCHEDULER_FRESH_BUILD: bool = False
    VLLM_OMNI_CACHE_NEG_PROMPT_EMBED: bool = False
    VLLM_OMNI_CACHE_PROMPT_EMBED: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Runtime Env Vars ==================
    # used in distributed environment to determine the master address
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    # used in distributed environment to manually set the communication port
    "MASTER_PORT": lambda: int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None,
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # ================== Host-side optimization flags ==================
    # Build a fresh inference scheduler per forward instead of deep-copying the
    # template each call (skips the per-forward copy.deepcopy of the scheduler).
    "VLLM_OMNI_SCHEDULER_FRESH_BUILD": lambda: os.environ.get("VLLM_OMNI_SCHEDULER_FRESH_BUILD", "0") == "1",
    # Cache the constant negative-prompt embeds (CFG uncond branch), keyed by device.
    "VLLM_OMNI_CACHE_NEG_PROMPT_EMBED": lambda: os.environ.get("VLLM_OMNI_CACHE_NEG_PROMPT_EMBED", "0") == "1",
    # Cache the positive (task) prompt encode per (prompt, device) in a small LRU.
    "VLLM_OMNI_CACHE_PROMPT_EMBED": lambda: os.environ.get("VLLM_OMNI_CACHE_PROMPT_EMBED", "0") == "1",
}


class PackagesEnvChecker:
    """Singleton class for checking package availability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        packages_info = {}
        packages_info["has_flash_attn"] = self._check_flash_attn()
        self.packages_info = packages_info

    def _check_flash_attn(self) -> bool:
        """Check if flash attention is available and compatible."""
        platform = current_omni_platform

        if platform.get_device_count() == 0:
            return False

        return platform.has_flash_attn_package()

    def get_packages_info(self) -> dict:
        """Get the packages info dictionary."""
        return self.packages_info


PACKAGES_CHECKER = PackagesEnvChecker()


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
