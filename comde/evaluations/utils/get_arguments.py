import pickle
import random
from functools import partial
from typing import Dict

import numpy as np

from comde.evaluations.modes.baselines import (
	evaluate_flatbc,
	evaluate_demogen,
	evaluate_promptdt,
	evaluate_bcz,
	evaluate_vima
)
from comde.evaluations.modes.comde_eval import evaluate_comde
from comde.utils.common.pretrained_forwards.jax_bert_base import bert_base_forward


def get_arguments(kwargs: Dict, mode: str, custom_seed: int):
	cfg = kwargs["cfg"]

	semantic_skills_sequence = kwargs.get("semantic_skills_sequence")
	param_for_skill = kwargs.get("param_for_skill")

	arguments = {
		"envs": kwargs["envs"],
		"save_results": cfg["save_results"],
	}

	info = dict()
	pretrained_models = kwargs["pretrained_models"]

	if mode == "flatbc":
		arguments.update({
			**kwargs["pretrained_models"],
			"semantic_skills_sequence": semantic_skills_sequence,
			"param_for_skill": param_for_skill,
		})

		info.update({"sequential_requirement": "not_used"})

	elif mode in [
		"demogen", "promptdt", "bcz", "vima", "flaxvima",
		"sensordemogen", "sensorbcz", "sensorflaxvima", "sensorsourcelanguagepromptdt"
	]:

		envs = kwargs["envs"]
		seq_req_mapping = envs[0].sequential_requirements_vector_mapping[cfg.sequential_requirement]
		n_envs = len(envs)
		str_seq_req = random.choice(list(seq_req_mapping.keys()))
		sequential_requirement = [seq_req_mapping[str_seq_req] for _ in range(n_envs)]
		random.seed(custom_seed)
		random.shuffle(sequential_requirement)

		sequential_requirement = np.array(sequential_requirement)
		non_functionalities = kwargs["non_functionalities"]
		param_for_skill = kwargs["param_for_skill"]

		source_skills_idxs = kwargs["source_skills_idx"]

		if mode in ["demogen", "bcz", "sensordemogen", "sensorbcz"]:
			if "sensor" not in mode:
				if mode == "demogen":
					with open(cfg["env"]["task_video_path"], "rb") as f:
						vl_feature_dict = pickle.load(f)

				elif mode == "bcz":
					if str(envs[0]) == "metaworld":
						vl_feature_dict = pretrained_models["baseline"].video_feature
					else:
						with open(cfg["env"]["task_video_path"], "rb") as f:
							vl_feature_dict = pickle.load(f)
				else:
					raise NotImplementedError()

				text_dict = pretrained_models["baseline"].episodic_inst
				source_video_embeddings = []
				for src_sk in source_skills_idxs:
					src_sk = " ".join([str(sk) for sk in src_sk])
					src_emb = vl_feature_dict.get(src_sk, list(text_dict[src_sk].values()))
					src_emb = random.choice(src_emb)
					source_video_embeddings.append(src_emb)

				source_video_embeddings = np.array(source_video_embeddings)

			elif "sensor" in mode:
				if mode in ["sensordemogen", "sensorbcz"]:
					with open(cfg["env"]["first_few_sensors_path"], "rb") as f:
						sensor_dict = pickle.load(f)

					source_obss = []
					source_acts = []
					for src_sk in source_skills_idxs:
						src_obs = [random.choice(sensor_dict[sk])["observations"] for sk in src_sk]
						src_act = [random.choice(sensor_dict[sk])["actions"] for sk in src_sk]

						source_obss.append(np.concatenate(src_obs, axis=0))
						source_acts.append(np.concatenate(src_act, axis=0))

					so = np.stack(source_obss, axis=0)	# [b, l, d]
					sa = np.stack(source_acts, axis=0)	# [b, l, d]
					batch_size = so.shape[0]
					# Actually not video, but just naming ...
					source_video_embeddings = np.concatenate((so, sa), axis=-1).reshape(batch_size, -1)

			else:
				raise NotImplementedError()

			arguments.update({
				**kwargs["pretrained_models"],
				"source_video_embeddings": source_video_embeddings,
				"sequential_requirement": sequential_requirement,
				"non_functionality": non_functionalities[:, 0, ...],
				"param_for_skill": param_for_skill
			})

		elif mode in ["vima", "flaxvima", "sensorflaxvima", "sensorsourcelanguagepromptdt"]:
			language_guidances = []
			for t, env in enumerate(envs):
				lg = env.get_language_guidance_from_template(
					sequential_requirement=cfg.sequential_requirement,
					non_functionality=cfg.non_functionality,
					source_skills_idx=source_skills_idxs[t],
					parameter=None,
					video_parsing=True
				)
				language_guidances.append(lg)
			source_skills_vecs = kwargs["source_skills_vec"]
			source_skills_vecs = np.array(source_skills_vecs)
			n_source_skills = np.array([source_skills_vecs.shape[1] for _ in range(n_envs)])
			model = pretrained_models["baseline"]
			rtgs = np.array([env.get_rtg() for env in envs])

			if mode == "vima":
				prompts, prompts_assets, prompts_maskings, prompts_assets_maskings = model.get_prompts_from_components(
					language_guidances=language_guidances,
					source_skills=source_skills_vecs,
					n_source_skills=n_source_skills,
					source_skills_idxs=source_skills_idxs
				)
				model.predict = partial(
					model.predict,
					prompt_assets=prompts_assets,
					prompt_assets_mask=prompts_assets_maskings
				)

			elif mode == "flaxvima":
				prompts, prompts_maskings = model.get_prompts_from_components(
					language_guidances=language_guidances,
					source_skills=source_skills_vecs,
					n_source_skills=n_source_skills,
					source_skills_idxs=source_skills_idxs
				)
				prompts = {"prompts": prompts}
				prompts_maskings = {"prompts_maskings": prompts_maskings}

			elif mode in ["sensorflaxvima", "sensorsourcelanguagepromptdt"]:
				prompts_dict = model.get_prompts_from_components(
					language_guidances=language_guidances,
					source_skills=source_skills_vecs,
					n_source_skills=n_source_skills,
					source_skills_idxs=source_skills_idxs
				)
				prompts = {
					"language_prompts": prompts_dict["language_prompts"],
					"sensor_prompts": prompts_dict["sensor_prompts"]
				}
				prompts_maskings = {
					"language_prompts_maskings": prompts_dict["language_prompts_maskings"],
					"sensor_prompts_maskings": prompts_dict["sensor_prompts_maskings"]
				}

				if mode == "sensorsourcelanguagepromptdt":
					rtgs = np.array([env.get_rtg() for env in envs])
					non_functionality = non_functionalities[:, 0, ...]
					arguments.update({
						"rtgs": rtgs,
						"sequential_requirement": sequential_requirement,
						"non_functionality": non_functionality
					})

			else:
				raise NotImplementedError()

			arguments.update({
				**kwargs["pretrained_models"],

				"prompts": prompts,
				"prompts_maskings": prompts_maskings,
				"param_for_skills": param_for_skill,
				"rtgs": rtgs,
			})

		else:
			envs = kwargs["envs"]
			prompts = []
			model = pretrained_models["baseline"]

			language_guidances = []
			for t, env in enumerate(envs):
				lg = env.get_language_guidance_from_template(
					sequential_requirement=cfg.sequential_requirement,
					non_functionality=cfg.non_functionality,
					source_skills_idx=source_skills_idxs[t],
					parameter=None,
					video_parsing=True
				)
				language_guidances.append(lg)

			if str(model) == "VLPromptDT":
				prompts, prompts_maskings = model.get_prompts_from_components(
					source_skills_idxs=source_skills_idxs,
					language_guidance=language_guidances
				)

			elif str(model) == "SourceLanguagePromptDT":
				qkv_info = bert_base_forward(language_guidances)
				prompts = qkv_info["language_embedding"]
				prompts_maskings = qkv_info["attention_mask"]

			elif str(model) == "TargetAllPromptDT":
				prompts.extend(semantic_skills_sequence)
				target_skills = np.array(semantic_skills_sequence)
				n_target_skills = np.array([sk.shape[0] for sk in semantic_skills_sequence]).reshape(-1, 1)
				batch_size = target_skills.shape[0]
				prompts_maskings = np.arange(target_skills.shape[1]).reshape(1, -1)  # [1, M]
				prompts_maskings = np.repeat(prompts_maskings, repeats=batch_size, axis=0)  # [b, M]
				prompts_maskings = np.where(prompts_maskings < n_target_skills, 1, 0)

			else:
				raise NotImplementedError("Undefined PromptDT")

			prompts = np.array(prompts)

			# Note: This raise an error if the number of source skills is different
			non_functionality = non_functionalities[:, 0, ...]

			rtgs = np.array([env.get_rtg() for env in envs])

			arguments.update({
				**kwargs["pretrained_models"],
				"prompts": prompts,
				"prompts_maskings": prompts_maskings,
				"sequential_requirement": sequential_requirement,
				"non_functionality": non_functionality,
				"param_for_skills": param_for_skill,
				"rtgs": rtgs,
			})

		info.update({"sequential_requirement": str_seq_req})

	elif mode == "comde":
		non_functionalities = kwargs["non_functionalities"]
		non_functionalities = np.zeros_like(non_functionalities)

		if cfg["use_optimal_target_skill"]:
			semantic_skills_sequence = semantic_skills_sequence
		else:
			semantic_skills_sequence = kwargs["pred_target_skills"]

		arguments.update({
			**kwargs["pretrained_models"],
			"target_skills": np.concatenate((semantic_skills_sequence, non_functionalities, param_for_skill), axis=-1),
			"use_optimal_next_skill": cfg["use_optimal_next_skill"],
			"termination_pred_interval": cfg["termination_pred_interval"]
		})
		info.update({"sequential_requirement": "not_used"})

	return arguments, info


def get_evaluation_function(kwargs: Dict, custom_seed: int):
	if "baseline" in kwargs["pretrained_models"]:
		model = kwargs["pretrained_models"]["baseline"]
		model_name = str(model)
		if model_name == "FlatBC":
			fn = evaluate_flatbc
			mode = model_name.lower()
		elif model_name == "DemoGen":
			fn = evaluate_demogen
			mode = model_name.lower()
		elif model_name in ["VLPromptDT", "SourceLanguagePromptDT", "TargetAllPromptDT"]:
			fn = evaluate_promptdt
			mode = "promptdt"
		elif model_name == "BCZ":
			fn = evaluate_bcz
			mode = model_name.lower()
		elif model_name == "VIMA":
			fn = evaluate_vima
			mode = model_name.lower()
		elif model_name == "FlaxVIMA":
			fn = evaluate_vima
			mode = model_name.lower()

		elif model_name == "SensorDemoGen":
			fn = evaluate_demogen
			mode = model_name.lower()
		elif model_name == "SensorBCZ":
			fn = evaluate_bcz
			mode = model_name.lower()
		elif model_name == "SensorFlaxVIMA":
			fn = evaluate_vima
			mode = model_name.lower()
		elif model_name == "SensorSourceLanguagePromptDT":
			fn = evaluate_vima
			mode = model_name.lower()

		else:
			raise NotImplementedError(f"Not implemented baseline: {str(model)}")
	else:
		mode = "comde"
		fn = evaluate_comde

	arguments, info = get_arguments(kwargs, mode, custom_seed)
	evaluation_function = partial(fn, **arguments)

	return evaluation_function, info
