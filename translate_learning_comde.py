from typing import Dict
from typing import Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from comde.rl.envs import get_dummy_env
from comde.trainer.compose_trainer import ComposeTrainer
from comde.utils.common.natural_languages.merge_tokens import merge_env_tokens


@hydra.main(version_base=None, config_path="config/train", config_name="comde_base.yaml")
def program(cfg: DictConfig) -> None:
	cfg = OmegaConf.to_container(cfg, resolve=True)  # type: Dict[str, Union[str, int, Dict]]

	assert cfg["mode"]["mode"] == "translate_learning", \
		f"Your mode is {cfg['mode']['mode']}. " \
		"Please add 'mode=translate_learning' to your command line if you want to train semantic skill translator"

	metaworld = get_dummy_env("metaworld", register_language_embedding=False, cfg=cfg["env"])
	kitchen = get_dummy_env("kitchen", register_language_embedding=False, cfg=cfg["env"])
	rlbench = get_dummy_env("rlbench", register_language_embedding=False, cfg=cfg["env"])

	envs = {
		"metaworld": metaworld,
		"kitchen": kitchen,
		"rlbench": rlbench
	}
	tokens, offset_info = merge_env_tokens(list(envs.values()))

	seq2seq = instantiate(cfg["translate_learner"], custom_tokens=tokens)
	seq2seq.offset_info = offset_info
	trainer = ComposeTrainer(cfg=cfg, envs=envs, seq2seq=seq2seq, offset_info=offset_info)

	for n_iter in range(cfg["max_iter"]):
		trainer.run()
		trainer.evaluate()


if __name__ == "__main__":
	program()
