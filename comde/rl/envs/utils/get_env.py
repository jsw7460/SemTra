import pickle
from pathlib import Path
from typing import Dict

from comde.rl.envs.metaworld.metaworld import DimFixedMetaWorld
from comde.rl.envs.utils.skill_to_vec import SkillToVec


def get_dummy_env(cfg: Dict) -> SkillToVec:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	env_name = cfg["name"]

	if "meta" in env_name.lower():
		env = DimFixedMetaWorld(seed=0, task=["box", "handle", "button", "door"])  # No meaning

	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	with open(Path(cfg["skill_to_vec_path"]), "rb") as f:
		skill_to_vec = pickle.load(f)

	env = SkillToVec(env=env, skill_to_vec=skill_to_vec)

	return env
