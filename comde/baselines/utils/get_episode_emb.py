import random
from typing import Dict
from functools import partial

import numpy as np

from comde.rl.buffers.type_aliases import ComDeBufferSample


def get_video_text_embeddings(
	video_feature_dict: Dict[str, np.ndarray],
	replay_data: ComDeBufferSample,
	text_dict: Dict[str, Dict[str, np.ndarray]] = None,
):
	"""
	:param video_feature_dict:
	:param text_dict:
		Episodic instruction: {"0 1 2": {
			'close the box, and then slide the puck into the goal, and then manipulate the handle': np.ndarray,
			'xxxx': np.ndarray,
			'yyyy': np.ndarray, ...
			}
		}
	:param replay_data:
	:return:
	"""
	src_skills_idxs = replay_data.source_skills_idxs
	tgt_skills_idxs = replay_data.target_skills_idxs
	n_src_skills = replay_data.n_source_skills
	n_tgt_skills = replay_data.n_target_skills

	source_video_embs = []
	target_video_embs = []

	video_found_idxs = []

	if text_dict is not None:
		source_text_embs = []
		target_text_embs = []

	# pad_xxx : zero padded components
	for t, (pad_src_sk, pad_tgt_sk, n_src, n_tgt) in enumerate(zip(src_skills_idxs, tgt_skills_idxs, n_src_skills, n_tgt_skills)):
		src_sk = pad_src_sk[:n_src]
		tgt_sk = pad_tgt_sk[:n_tgt]

		src_sk = " ".join([str(sk) for sk in src_sk])
		tgt_sk = " ".join([str(sk) for sk in tgt_sk])

		src_embs = video_feature_dict.get(src_sk, list(text_dict[src_sk].values()))
		tgt_embs = video_feature_dict.get(tgt_sk, list(text_dict[tgt_sk].values()))

		if src_sk in video_feature_dict:
			video_found_idxs.append(t)

		src_emb = random.choice(src_embs)
		tgt_emb = random.choice(tgt_embs)

		source_video_embs.append(src_emb)
		target_video_embs.append(tgt_emb)

		if text_dict is not None:
			src_text_emb = random.choice(list(text_dict[src_sk].values()))
			tgt_text_emb = random.choice(list(text_dict[tgt_sk].values()))

			source_text_embs.append(src_text_emb)
			target_text_embs.append(tgt_text_emb)

	source_video_embs = np.array(source_video_embs)
	target_video_embs = np.array(target_video_embs)

	emb_info = {
		"source_video_embeddings": source_video_embs,
		"target_video_embeddings": target_video_embs,
		"video_found_idxs": video_found_idxs
	}
	if text_dict is not None:
		source_text_embs = np.array(source_text_embs)
		target_text_embs = np.array(target_text_embs)
		emb_info.update({"source_text_embeddings": source_text_embs, "target_text_embs": target_text_embs})

	return emb_info


def get_sensor_text_embeddings(
	sensor_dict: Dict[str, np.ndarray],
	replay_data: ComDeBufferSample,
	text_dict: Dict[str, Dict[str, np.ndarray]] = None,
):
	"""
	:param sensor_dict:
	:param text_dict:
		Episodic instruction: {"0 1 2": {
			'close the box, and then slide the puck into the goal, and then manipulate the handle': np.ndarray,
			'xxxx': np.ndarray,
			'yyyy': np.ndarray, ...
			}
		}
	:param replay_data:
	:return:
	"""
	src_skills_idxs = replay_data.source_skills_idxs
	tgt_skills_idxs = replay_data.target_skills_idxs
	n_src_skills = replay_data.n_source_skills
	n_tgt_skills = replay_data.n_target_skills

	source_obss = []
	source_acts = []
	target_obss = []
	target_acts = []

	source_text_embs = []
	target_text_embs = []

	# pad_xxx : zero padded components
	for t, (pad_src_sk, pad_tgt_sk, n_src, n_tgt) in enumerate(zip(src_skills_idxs, tgt_skills_idxs, n_src_skills, n_tgt_skills)):
		src_sk = pad_src_sk[:n_src]
		tgt_sk = pad_tgt_sk[:n_tgt]

		src_obs = [random.choice(sensor_dict[sk])["observations"] for sk in src_sk]
		src_act = [random.choice(sensor_dict[sk])["actions"] for sk in src_sk]

		source_obss.append(np.concatenate(src_obs, axis=0))
		source_acts.append(np.concatenate(src_act, axis=0))

		tgt_obs = [random.choice(sensor_dict[sk])["observations"] for sk in src_sk]
		tgt_act = [random.choice(sensor_dict[sk])["actions"] for sk in src_sk]

		target_obss.append(np.concatenate(tgt_obs, axis=0))
		target_acts.append(np.concatenate(tgt_act, axis=0))

		if text_dict is not None:
			src_sk = " ".join([str(sk) for sk in src_sk])
			tgt_sk = " ".join([str(sk) for sk in tgt_sk])
			src_text_emb = random.choice(list(text_dict[src_sk].values()))
			tgt_text_emb = random.choice(list(text_dict[tgt_sk].values()))

			source_text_embs.append(src_text_emb)
			target_text_embs.append(tgt_text_emb)

	source_observations = np.stack(source_obss, axis=0)
	source_actions = np.stack(source_acts, axis=0)
	target_observations = np.stack(target_obss, axis=0)
	target_actions = np.stack(target_acts, axis=0)

	emb_info = {
		"source_observations": source_observations,
		"source_actions": source_actions,
		"target_observations": target_observations,
		"target_actions": target_actions
	}
	if text_dict is not None:
		source_text_embs = np.array(source_text_embs)
		target_text_embs = np.array(target_text_embs)
		emb_info.update({"source_text_embeddings": source_text_embs, "target_text_embs": target_text_embs})

	return emb_info


get_video_embeddings = partial(get_video_text_embeddings, text_dict=None)