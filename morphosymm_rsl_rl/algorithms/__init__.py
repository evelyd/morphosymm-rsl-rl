# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .ppo_symm_data_augment import PPOSymmDataAugmented
from .ppo import PPO
from .ppo_dae_online import PPODAEOnline

try:
	from .ppo_symm_dae_online import PPOSymmDAEOnline
except ImportError:
	PPOSymmDAEOnline = None

try:
	from .ppo_rff import PPORFF
except ImportError:
	PPORFF = None

try:
	from .ppo_symm_erff import PPOSymmERFF
except ImportError:
	PPOSymmERFF = None

__all__ = ["PPOSymmDataAugmented", "PPO", "PPODAEOnline", "PPOSymmDAEOnline", "PPORFF", "PPOSymmERFF"]
