import numpy as np
from comde.rl.buffers.type_aliases import ComDeBufferSample


def get_episodic_level_skills(replay_data: ComDeBufferSample, param_repeats: int = 1, n_target_skill: int = None):
	"""
	Return episode level skills
	This is different from timestep level skills
	i.e., target_skills = ["skill A", "skill D", "skill B"] (also parameterized)
		-> return: concatenation of them.
	"""

	source_skills = replay_data.source_skills
	target_skills = replay_data.target_skills

	if n_target_skill is None:
		n_target_skill = target_skills.shape[1]

	batch_source_params = []
	batch_target_params = []

	non_func = replay_data.non_functionality
	source_parameter = replay_data.source_parameters
	parameter = replay_data.parameters

	source_skills_idxs = replay_data.source_skills_idxs[:, :n_target_skill, ...]
	target_skills_idxs = replay_data.target_skills_idxs[:, :n_target_skill, ...]
	for i, (source_idx, target_idx) in enumerate(zip(source_skills_idxs, target_skills_idxs)):
		# i: loop over batch
		source_params = [source_parameter[i][idx] for idx in source_idx]
		target_params = [parameter[i][idx] for idx in target_idx]
		batch_source_params.append(source_params)
		batch_target_params.append(target_params)

	batch_source_params = np.array(batch_source_params)[..., np.newaxis]  # b, n_source_skill
	batch_target_params = np.array(batch_target_params)[..., np.newaxis]  # b, n_target_skill
	batch_source_params = np.repeat(batch_source_params, axis=-1, repeats=param_repeats)	# b, n_source_skills, param_dim
	batch_target_params = np.repeat(batch_target_params, axis=-1, repeats=param_repeats)	# b, n_target_skills, param_dim
	batch_size = source_skills.shape[0]

	source_skills = source_skills[:, :n_target_skill, ...]  # [b, n, d]
	target_skills = target_skills[:, :n_target_skill, ...]  # [b, n, d]

	skill_dim = source_skills.shape[-1]
	param_dim = batch_source_params.shape[-1]
	parameterized_source_skills = np.zeros((batch_size, n_target_skill, skill_dim + param_dim))
	parameterized_target_skills = np.zeros((batch_size, n_target_skill, skill_dim + param_dim))
	parameterized_source_skills[..., :skill_dim] = source_skills
	parameterized_source_skills[..., skill_dim:] = batch_source_params
	# parameterized_source_skills = np.concatenate((source_skills, batch_source_params), axis=-1)

	parameterized_target_skills[..., :skill_dim] = target_skills
	parameterized_target_skills[..., skill_dim:] = batch_target_params
	# parameterized_target_skills = np.concatenate((target_skills, batch_target_params), axis=-1)

	flat_parameterized_source_skills = parameterized_source_skills.reshape(batch_size, -1)
	flat_parameterized_target_skills = parameterized_target_skills.reshape(batch_size, -1)

	flat_nonfunc_parameterized_source_skills = np.concatenate((flat_parameterized_source_skills, non_func), axis=-1)
	flat_nonfunc_parameterized_target_skills = np.concatenate((flat_parameterized_target_skills, non_func), axis=-1)

	info = {
		"flat_nonfunc_parameterized_source_skills": flat_nonfunc_parameterized_source_skills,  # [b, d]
		"flat_nonfunc_parameterized_target_skills": flat_nonfunc_parameterized_target_skills,  # [b, d]
		"parameterized_target_skills": parameterized_target_skills,  # [b, n_target_skill, d]
		"param_for_source_skills": batch_source_params,
		"param_for_target_skills": batch_target_params,
	}
	return info
