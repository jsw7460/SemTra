import os
import pickle
import random
from itertools import product
from typing import NamedTuple, Any

import h5py
import numpy as np


class Template(NamedTuple):
	sequential_requirement: str
	non_functionality: str
	parameter: Any


file = h5py.File("/home/jsw7460/mnt/comde_datasets/metaworld/speed/3_target_skills/51/data128064.hdf5")
wind_file = h5py.File("/home/jsw7460/mnt/comde_datasets/metaworld/wind/3_target_skills/0/data0.hdf5")

# Make keys: [actions, non_functionality, observations, parameter, sequential_requirement, skills_done, skills_idxs, skills_order, source_skills, target_skills]
WINDS = [
	-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
	0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35
]

KITCHEN_SKILL_INDICES = {
	'bottom burner': 0,
	'top burner': 1,
	'light switch': 2,
	'slide cabinet': 3,
	'hinge cabinet': 4,
	'microwave': 5,
	'kettle': 6,
}


def get_wind_templates():
	easy_tasks = {0, 1, 2, 3, 4, 5, 6}
	templates = []
	for wind in WINDS:
		parameter = {k: wind for k in easy_tasks}
		template = Template(
			sequential_requirement="X",
			non_functionality="wind",
			parameter=parameter
		)
		templates.append(template)

	return templates


def template_split_target_to_source(target_skills: list, operators: list, template: NamedTuple) -> list:
	"""
	split target skills to any possible source skills combination and
	return them with corresponding operator

	operators : [sequential, reverse, replace]
	skills : puck(1), drawer(3), button(4), door(6)

	ex.
	ret = [
		{
		'video1': [1],
		'video2': [3, 4, 6],
		'operator': 'sequential'
		},
		{
		'video1': [6, 4],
		'video2': [3],
		'video3': [1],
		'operator': 'reverse'
		},
		...
	]
	"""

	def split(l: list, r: list, split_num: int, src_skills: list):
		if split_num == 1:
			l.append(r)
			src_skills.append(l)
		elif split_num == len(r):
			for skill in r:
				l.append([skill])
			src_skills.append(l)
		else:
			for idx in range(0, len(r) - split_num + 1):
				tmp = l.copy()
				tmp.append(r[:idx + 1])
				split(tmp, r[idx + 1:], split_num - 1, src_skills)

	def split_target_skills(tar_skills: list):
		"""
		split target skills to any possible source skills combination
		ex.
		target_skills = [1, 2, 3]
		ret = [
			[[1, 2, 3]]
			[[1], [2, 3]]
			[[1, 2], [3]]
			[[1], [2], [3]]
		]
		"""
		src_skills = []
		for split_num in range(1, len(tar_skills) + 1):
			split([], tar_skills, split_num, src_skills)
		return src_skills

	def combine_with_operator(skills: list, operator: str, sources: list, template: NamedTuple):
		for src_skill in skills:
			src = {}
			video_cnt = 1
			template = template._replace(sequential_requirement=operator)
			for skill in src_skill:
				src[f'video{video_cnt}'] = skill
				video_cnt += 1
			src['template'] = template
			sources.append(src)

	replace_skills = set([0, 1, 2, 3, 4, 5, 6]) - set(target_skills)
	sources = []

	if 'sequential' in operators:
		sequential_skills = split_target_skills(target_skills)
		sequential_skills = [sequential_skills[0]]
		combine_with_operator(sequential_skills, 'sequential', sources, template)

	if 'reverse' in operators:
		reverse_skills = split_target_skills(target_skills[::-1])
		reverse_skills = [reverse_skills[0]]
		combine_with_operator(reverse_skills, 'reverse', sources, template)

	if 'replace' in operators and len(replace_skills) > 0:
		cpy = sources.copy()
		for _ in range(len(list(product(target_skills, replace_skills))) - 1):
			for e in cpy:
				sources.append(e)

		for t_skill, s_skill in product(target_skills, replace_skills):
			cpy = target_skills.copy()
			for idx, value in enumerate(cpy):
				if value == t_skill:
					cpy[idx] = s_skill

			replace_skills = split_target_skills(cpy)
			replace_skills = [replace_skills[0]]
			replace_operator = f'replace {s_skill} with {t_skill}'

			combine_with_operator(replace_skills, replace_operator, sources, template)

	random.shuffle(sources)
	return sources


if __name__ == "__main__":

	operators = ["sequential", "reverse", "replace"]

	with open("/home/jsw7460/0419_kitchen/kitchen_skill_appended_Total_2.pkl", "rb") as f:
		dataset = pickle.load(f)

	file_idx = 0
	folder_idx = 0
	save_path = f'/home/jsw7460/mnt/comde_datasets/kitchen/wo_wind/4_target_skills'
	# save_path = f'/home/jsw7460/foo'
	folder_path = save_path + '/' + f'{folder_idx}'
	os.makedirs(folder_path, exist_ok=True)

	d_observations = np.array(dataset["observations"])
	d_actions = np.array(dataset["actions"])
	d_rewards = np.array(dataset["rewards"])
	d_terminals = np.array(dataset["terminals"])
	d_skills = np.array(dataset["skills"])
	d_skill_done = np.array(dataset["skill_done"])
	d_infos = dataset["infos"]
	# d_skill_feat_map = np.array(dataset["skill_feat_map"])

	no_wind_indices = []
	for t, info in enumerate(d_infos):
		noise = info["action_noise"]
		if noise.sum() == 0:
			no_wind_indices.append(t)
	no_wind_indices = np.array(no_wind_indices, dtype=np.int32)

	d_observations = d_observations[no_wind_indices]
	d_actions = d_actions[no_wind_indices]
	d_rewards = d_rewards[no_wind_indices]
	d_terminals = d_terminals[no_wind_indices]
	d_skills = d_skills[no_wind_indices]
	d_skill_done = d_skill_done[no_wind_indices]
	fake_infos = []
	for idx in no_wind_indices:
		fake_infos.append(d_infos[idx])

	d_infos = fake_infos
	# d_skill_feat_map = d_skill_feat_map[no_wind_indices]

	assert len(d_observations) == len(d_actions) == len(d_rewards) == len(d_terminals) == len(d_skills) == len(
		d_skill_done) == len(d_infos)

	prev_done = 0
	for t in range(len(d_observations)):
		if d_terminals[t]:
			print(f"Current {t} / {len(d_observations)}")
			observations = d_observations[prev_done: t + 1].copy()
			actions = d_actions[prev_done: t + 1].copy()
			terminals = d_terminals[prev_done: t + 1].copy()
			skills_done = d_skill_done[prev_done: t + 1].copy()
			skills_idxs = d_skills[prev_done: t + 1].copy()

			_target_skills = d_infos[prev_done]["skill_seq"]
			target_skills = []
			for j in range(len(_target_skills)):
				name = _target_skills[j]
				target_skills.append(KITCHEN_SKILL_INDICES[name])

			prev_name = skills_idxs[0]
			skills_order = []
			order = 0
			for j in range(len(skills_idxs)):
				name = skills_idxs[j]
				skills_idxs[j] = int(KITCHEN_SKILL_INDICES[name])
				skills_order.append(order)
				if skills_done[j]:
					order += 1
			prev_done = t + 1

			skills_idxs = skills_idxs.astype("i4")
			# ===== Make source skills
			templates = get_wind_templates()
			for template in templates:
				wind_mean = list(template.parameter.values())[0]
				for v in template.parameter.values():
					assert v == wind_mean

				if wind_mean != 0.0:
					continue

				wind = np.array([wind_mean, 0., 0., 0., 0., 0., 0., 0., 0.]).reshape(1, -1)
				source_skills = template_split_target_to_source(target_skills, operators, template)

				# for _ in source_skills:
				# 	print(_)
				# exit()

				for source_skill in source_skills:
					wind_appended_actions = actions - wind

					# exit()
					data_path = os.path.join(folder_path, f'data{file_idx}.hdf5')
					# data_path = "/home/jsw7460/foo.hdf5"
					with h5py.File(data_path, 'w') as f:
						# source_skill, operator
						for k, v in source_skill.items():
							if 'video' in k:
								f['source_skills/' + k] = v
							elif 'template' in k:
								f.create_dataset('sequential_requirement', data=v.sequential_requirement)
								f.create_dataset('non_functionality', data=v.non_functionality)
								f.create_dataset('parameter', data=str(v.parameter))
							else:
								raise ValueError()

						# target_skill
						# print("SKILLS IDXS", skills_idxs)
						f.create_dataset('target_skills', data=target_skills)

						# obs, action, skill_idx
						f.create_dataset('observations', data=observations)
						f.create_dataset('actions', data=wind_appended_actions)
						f.create_dataset('skills_idxs', data=skills_idxs)
						f.create_dataset('skills_done', data=skills_done)
						f.create_dataset('skills_order', data=skills_order)

					file_idx += 1

					if file_idx % 100000 == 0:
						folder_idx += 1
						folder_path = save_path + '/' + f'{folder_idx}'
						os.makedirs(folder_path, exist_ok=True)

			print("\n\n\n\n\n\n\n\n\n\n\n")
