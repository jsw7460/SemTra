import pickle
import random
from functools import partial
from typing import Dict

import numpy as np

from comde.evaluations.modes.baselines import (evaluate_bcz, evaluate_demogen,
                                               evaluate_flatbc,
                                               evaluate_promptdt,
                                               evaluate_vima)
from comde.evaluations.modes.comde_eval import evaluate_comde


def get_arguments(kwargs: Dict, mode: str, custom_seed: int):
	cfg = kwargs["cfg"]

	semantic_skills_sequence = kwargs.get("semantic_skills_sequence")
	param_for_skill = kwargs.get("param_for_skill")

	arguments = {
		"envs": kwargs["envs"],
		"save_results": cfg["save_results"],
	}

	info = dict()

	if mode == "flatbc":
		arguments.update({
			**kwargs["pretrained_models"],
			"semantic_skills_sequence": semantic_skills_sequence,
			"param_for_skill": param_for_skill,
		})

		info.update({"sequential_requirement": "not_used"})

	elif mode in ["demogen", "promptdt", "bcz"]:
		pretrained_cfg = kwargs["pretrained_cfg"]
		pretrained_models = kwargs["pretrained_models"]
		seq_req_path = pretrained_cfg["env"]["sequential_requirements_path"]

		with open(seq_req_path, "rb") as f:
			seq_req_mapping = pickle.load(f)
			seq_req_mapping = seq_req_mapping[cfg.sequential_requirement]

		n_envs = len(kwargs["envs"])
		str_seq_req = random.choice(list(seq_req_mapping.keys()))
		sequential_requirement = [seq_req_mapping[str_seq_req] for _ in range(n_envs)]
		random.seed(custom_seed)
		random.shuffle(sequential_requirement)

		sequential_requirement = np.array(sequential_requirement)
		non_functionalities = kwargs["non_functionalities"]
		param_for_skill = kwargs["param_for_skill"]

		source_skills_vecs = kwargs["source_skills_vec"]
		source_skills_idxs = kwargs["source_skills_idx"]

		if mode in ["demogen", "bcz"]:
			if mode == "demogen":
				vl_feature_dict = pretrained_models["baseline"].video_feature_dict
			else:
				vl_feature_dict = pretrained_models["baseline"].video_feature
			# This is for the case.
			text_dict = pretrained_models["baseline"].episodic_inst
			source_video_embeddings = []
			for src_sk in source_skills_idxs:
				src_sk = " ".join([str(sk) for sk in src_sk])
				src_emb = vl_feature_dict.get(src_sk, list(text_dict[src_sk].values()))
				src_emb = random.choice(src_emb)
				source_video_embeddings.append(src_emb)

			source_video_embeddings = np.array(source_video_embeddings)
			arguments.update({
				**kwargs["pretrained_models"],
				"source_video_embeddings": source_video_embeddings,
				"sequential_requirement": sequential_requirement,
				"non_functionality": non_functionalities[:, 0, ...],
				"param_for_skill": param_for_skill
			})
		else:
			envs = kwargs["envs"]
			prompts = []
			model = pretrained_models["baseline"]

			if str(model) == "VLPromptDT":
				firstimage_mapping = pretrained_models["baseline"].firstimage_mapping
				for env, source_skill_idxs in zip(envs, source_skills_idxs):
					tmp_prompts = np.array([random.choice(firstimage_mapping[str(sk)]) for sk in source_skill_idxs])
					prompts.append(tmp_prompts)

				prompts_maskings = None

			elif str(model) == "SourceLanguagePromptDT":
				prompts.extend(source_skills_vecs)
				source_skills = np.array(source_skills_vecs)
				n_source_skills = np.array([sk.shape[0] for sk in source_skills_vecs]).reshape(-1, 1)
				batch_size = source_skills.shape[0]
				prompts_maskings = np.arange(source_skills.shape[1]).reshape(1, -1)  # [1, M]
				prompts_maskings = np.repeat(prompts_maskings, repeats=batch_size, axis=0)  # [b, M]
				prompts_maskings = np.where(prompts_maskings < n_source_skills, 1, 0)

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
		if str(model) == "FlatBC":
			fn = evaluate_flatbc
			mode = "flatbc"
		elif str(model) == "DemoGen":
			fn = evaluate_demogen
			mode = "demogen"
		elif str(model) in ["VLPromptDT", "SourceLanguagePromptDT", "TargetAllPromptDT"]:
			fn = evaluate_promptdt
			mode = "promptdt"
		elif str(model) == "BCZ":
			fn = evaluate_bcz
			mode = "bcz"
		elif str(model) == "VIMA":
			fn = evaluate_vima
			mode = "vima"
		else:
			raise NotImplementedError(f"Not implemented baseline: {str(model)}")

	else:
		mode = "comde"
		fn = evaluate_comde

	arguments, info = get_arguments(kwargs, mode, custom_seed)
	evaluation_function = partial(fn, **arguments)

	return evaluation_function, info
