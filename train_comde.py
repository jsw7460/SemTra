# from jax.config import config
#
# config.update("jax_debug_nans", True)

import random
from copy import deepcopy
from typing import Dict, Union

from comde.trainer.baseline_trainer import BaselineTrainer

random.seed(7)

import hydra
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from comde.rl.buffers import ComdeBuffer
from comde.rl.envs import get_dummy_env
from comde.utils.common.load_data_paths import load_data_paths
from comde.utils.common.normalization import get_observation_statistics


@hydra.main(version_base=None, config_path="config/train", config_name="vima.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	data_dirs, hdf_files = load_data_paths(cfg)

	print(f"This program uses {len(hdf_files)} trajectories for the training.")
	cfg["n_trained_trajectory"] = len(hdf_files)
	dataset_window_size = len(hdf_files) // len(data_dirs)

	if cfg["state_normalization"]:
		statistics = get_observation_statistics(data_dirs)
		low_policy_cfgs = cfg["low_policy"]  # type: Dict
		low_policy_cfgs["cfg"].update({**statistics})

	env_name = cfg["env"]["name"].lower()
	env = get_dummy_env(env_name, cfg["env"])  # Dummy env for obtain an observation and action space.

	modules_dict = {}
	for module in cfg["modules"]:
		if module == "seq2seq":
			cfg[module].update({"custom_tokens": env.skill_infos})
		modules_dict[module] = instantiate(cfg[module])

	trainer_cls = get_class(cfg["trainer"])
	if trainer_cls == BaselineTrainer:
		trainer = trainer_cls(
			cfg=cfg,
			env=env,
			baseline=modules_dict["baseline"],
			skill_infos=env.skill_infos
		)
	else:
		trainer = trainer_cls(
			cfg=cfg,
			env=env,
			**modules_dict
		)

	for n_iter in range(cfg["max_iter"]):
		n_iter = (n_iter % len(data_dirs))
		trajectories = hdf_files[n_iter * dataset_window_size: (n_iter + 1) * dataset_window_size]
		random.shuffle(trajectories)

		replay_buffer = ComdeBuffer(
			env=env,
			subseq_len=cfg["subseq_len"],
			cfg=cfg["dataset"]
		)
		replay_buffer.add_episodes_from_h5py(
			paths={"trajectory": trajectories[: -10]},
			sequential_requirements_mapping=deepcopy(env.sequential_requirements_vector_mapping),
			non_functionalities_mapping=deepcopy(env.non_functionalities_vector_mapping),
			# guidance_to_prm=pretrained_modules["prompt_learner"]
		)
		trainer.run(replay_buffer)
		eval_buffer = ComdeBuffer(
			env=env,
			subseq_len=cfg["subseq_len"],
			cfg=cfg["dataset"]
		)
		eval_buffer.add_episodes_from_h5py(
			paths={"trajectory": trajectories[-10:]},
			sequential_requirements_mapping=deepcopy(env.sequential_requirements_vector_mapping),
			non_functionalities_mapping=deepcopy(env.non_functionalities_vector_mapping),
			# guidance_to_prm=pretrained_modules["prompt_learner"]
		)
		trainer.evaluate(eval_buffer)


if __name__ == "__main__":
	program()
