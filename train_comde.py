import random

random.seed(7)

from copy import deepcopy
from typing import Dict, Union

from comde.utils.common.load_data_paths import load_data_paths

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

from comde.rl.buffers import ComdeBuffer
from comde.rl.envs import get_dummy_env
from comde.utils.common.normalization import get_observation_statistics


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	env_name = cfg["env"]["name"].lower()
	# Dummy env for obtain an observation and action space.
	env = get_dummy_env(env_name, cfg["env"], register_language_embedding=False)
	hdf_files, hdf_dirs = load_data_paths(cfg, env, rm_eval_tasks=False)

	dataset_window_size = len(hdf_files) // len(hdf_dirs)

	if cfg["state_normalization"]:
		raise NotImplementedError("Obsolete")
		statistics = get_observation_statistics(data_dirs)
		low_policy_cfgs = cfg["low_policy"]  # type: Dict
		low_policy_cfgs["cfg"].update({**statistics})

	modules_dict = {}
	for module in cfg["modules"]:
		if module == "seq2seq":
			cfg[module].update({"custom_tokens": env.skill_infos})
		modules_dict[module] = instantiate(cfg[module])

	trainer_cls = get_class(cfg["trainer"])
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		**modules_dict
	)

	for n_iter in range(cfg["max_iter"]):
		data_start = n_iter % len(hdf_dirs)
		trajectories = hdf_files[data_start * dataset_window_size: (data_start + 1) * dataset_window_size]
		random.shuffle(trajectories)

		replay_buffer = ComdeBuffer(
			env=env,
			subseq_len=cfg["subseq_len"],
			cfg=cfg["dataset"]
		)
		train_available = replay_buffer.add_episodes_from_h5py(
			paths={"trajectory": trajectories[: -10]},
			sequential_requirements_mapping=deepcopy(env.sequential_requirements_vector_mapping),
			non_functionalities_mapping=deepcopy(env.non_functionalities_vector_mapping),
		)
		eval_buffer = ComdeBuffer(
			env=env,
			subseq_len=cfg["subseq_len"],
			cfg=cfg["dataset"]
		)
		eval_available = eval_buffer.add_episodes_from_h5py(
			paths={"trajectory": trajectories[-10:]},
			sequential_requirements_mapping=deepcopy(env.sequential_requirements_vector_mapping),
			non_functionalities_mapping=deepcopy(env.non_functionalities_vector_mapping),
		)

		if train_available:
			trainer.run(replay_buffer)
		if eval_available:
			trainer.evaluate(eval_buffer)


if __name__ == "__main__":
	program()
