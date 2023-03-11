import pickle
from pathlib import Path
from typing import Dict

from comde.rl.envs.metaworld.metaworld import DimFixedMetaWorld
from comde.rl.envs.utils.idx_to_skills import IdxToSkill


def get_dummy_env(cfg: Dict) -> IdxToSkill:
	"""
	Note: This is not responsible for evaluate the env.
	"""
	env_name = cfg["name"]

	if "meta" in env_name.lower():
		env = DimFixedMetaWorld(seed=0, task=["box", "handle", "button", "door"])  # No meaning

	else:
		raise NotImplementedError(f"Not supported: {env_name}")

	with open(Path(cfg["idx_to_skills_path"]), "rb") as f:
		idx_to_skills = pickle.load(f)

	env = IdxToSkill(env=env, idx_to_skills=idx_to_skills)

	return env
