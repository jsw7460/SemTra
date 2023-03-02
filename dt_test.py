from jax.config import config

config.update("jax_debug_nans", True)

import pickle

import d4rl
import gym
import hydra
from omegaconf import DictConfig, OmegaConf
from comde.comde_modules.low_policies import DecisionTransformer
from comde.rl.buffers import EpisodicMaskingBuffer
from comde.rl.envs import TimeLimitEnv
from comde.evaluations import evaluate_primdt
from comde.utils.common.normalization import get_normalized_data

import wandb

_ = d4rl.infos

TARGET_RTGS = {
	"hopper": [3600, 1800],
	"halfcheetah": [12000, 6000],
	"walker2d": [5000, 2500]
}


@hydra.main(version_base=None, config_path="config/bc", config_name="dt")
def program(cfg: DictConfig) -> None:
	env_name = cfg["env_name"]
	if "bullet" not in env_name:
		env_name = f"bullet-{env_name}"
	if "v0" not in env_name:
		env_name = f"{env_name}-v0"

	env = gym.make(env_name)
	env = TimeLimitEnv(env, limit=cfg["env_time_limit"])

	cfg.update({
		"observation_dim": env.observation_space.shape[-1],
		"action_dim": env.action_space.shape[-1]
	})

	with open(cfg["dataset_path"], "rb") as f:
		dataset = pickle.load(f)
		dataset, normalization_dict = get_normalized_data(dataset, cfg["data_normalization"])

	cfg = OmegaConf.to_container(cfg, resolve=True)
	wandb.init(
		project="v5_kitchen_vq_hidden",
		entity=cfg["entity_name"],
		name=cfg["run_name"],
		config=cfg
	)

	model = DecisionTransformer(seed=cfg["seed"], cfg=cfg["dt_config"])
	training_buffer = EpisodicMaskingBuffer(
		observation_space=env.observation_space,
		action_space=env.action_space,
		subseq_len=cfg["subseq_len"],
		buffer_size=len(dataset["observations"])
	)
	training_buffer.add_dict_chunk(dataset)

	if "hopper" in cfg["env_name"].lower():
		rtgs = TARGET_RTGS["hopper"]
	elif "halfcheetah" in cfg["env_name"].lower():
		rtgs = TARGET_RTGS["halfcheetah"]
	elif "walker2d" in cfg["env_name"].lower():
		rtgs = TARGET_RTGS["walker2d"]
	else:

		raise NotImplementedError("Undefined env")

	for it in range(cfg["max_iters"]):
		for step in range(cfg["num_steps_per_iter"]):
			replay_data = training_buffer.sample(batch_size=cfg["batch_size"])
			model.update(replay_data)
			if (step % 1000) == 0:
				model.dump_logs(it * cfg["num_steps_per_iter"] + step)

		episodic_info, wandb_info = evaluate_primdt(
			env=env,
			dt=model,
			init_rtgs=rtgs,
			normalization_dict=normalization_dict
		)

		for k, v in episodic_info.items():
			print(f"Env: {env_name}", k, sum(v["rewards"]))

if __name__ == "__main__":
	program()
