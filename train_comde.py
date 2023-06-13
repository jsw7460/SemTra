from jax.config import config

config.update("jax_debug_nans", True)

import random
from typing import Dict, Union, Type
from comde.utils.interfaces.i_savable.i_savable import IJaxSavable

random.seed(7)

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

	pretrained_modules = {}
	for prtr in cfg["mode"]["pretrained_modules"]:
		prtr_module = prtr["module"]
		prtr_path = prtr["path"]
		module_cls = get_class(cfg[prtr_module]["_target_"])  # type: Union[type, Type[IJaxSavable]]
		module_instance = module_cls.load(prtr_path)
		pretrained_modules[prtr_module] = module_instance

	data_dirs, hdf_files = load_data_paths(cfg)

	print(f"This program uses {len(hdf_files)} trajectories for the training.")
	dataset_window_size = len(hdf_files) // len(data_dirs)

	if cfg["state_normalization"]:
		statistics = get_observation_statistics(data_dirs)
		low_policy_cfgs = cfg["low_policy"]  # type: Dict
		low_policy_cfgs["cfg"].update({**statistics})

	env = get_dummy_env(cfg["env"])  # Dummy env for obtain an observation and action space.
	modules_dict = {module: instantiate(cfg[module]) for module in cfg["modules"]}

	trainer_cls = get_class(cfg["trainer"])

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
			paths={
				"trajectory": trajectories[: -10],
				"sequential_requirements": cfg["sequential_requirements_path"],
				"non_functionalities": cfg["non_functionalities_path"]
			},
			guidance_to_prm=pretrained_modules["prompt_learner"]
		)
		trainer.run(replay_buffer)
		eval_buffer = ComdeBuffer(
			env=env,
			subseq_len=cfg["subseq_len"],
			cfg=cfg["dataset"]
		)
		eval_buffer.add_episodes_from_h5py(
			paths={
				"trajectory": trajectories[-10:],
				"sequential_requirements": cfg["sequential_requirements_path"],
				"non_functionalities": cfg["non_functionalities_path"]
			}
		)
		trainer.evaluate(eval_buffer)


if __name__ == "__main__":
	program()
