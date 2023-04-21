import pickle
from itertools import permutations


onehot_skills_mapping = {
		'bottom_burner': 0,
		'top_burner': 1,
		'light_switch': 2,
		'slide_cabinet': 3,
		'hinge_cabinet': 4,
		'microwave': 5,
		'kettle': 6,
	}

idx2skills = {str(v): k for k, v in onehot_skills_mapping.items()}

sequential_requirements = ["sequential", "reverse"]
idx_permutations = list(permutations([i for i in range(7)], 2))
for (i, j) in idx_permutations:
	fmt = f"replace {i} with {j}"
	sequential_requirements.append(fmt)

skills = list(onehot_skills_mapping.keys())


# with open("/home/jsw7460/mnt/comde_datasets/tasks_for_evals/kitchen/eval-9", "rb") as f:
# 	tasks_for_eval = pickle.load(f)

n_target_skills = 3
r = n_target_skills
nPr = permutations(skills, r)

optimal_source_skills = dict()

for target_skills in list(nPr):
	l_target_skills = list(target_skills)
	for i in range(len(l_target_skills)):
		l_target_skills[i] = l_target_skills[i].replace("_", " ")

	target_skills = " ".join(target_skills)	# list

	optimal_source_skills[target_skills] = dict()
	for req in sequential_requirements:
		if req == "sequential":
			optimal_source_skills[target_skills][req] = l_target_skills.copy()

		elif req == "reverse":
			optimal_source_skills[target_skills][req] = l_target_skills.copy()[::-1]

		else:
			# Replace i with j
			i = req.split("with")[0].strip()[-1]
			j = req.split("with")[1].strip()[0]

			skill_from = idx2skills[i].replace("_", " ")
			skill_to = idx2skills[j].replace("_", " ")

			if (skill_to not in l_target_skills) or (skill_from in l_target_skills):
				optimal_source_skills[target_skills][req] = None

			else:
				replaced_l_target_skills = []
				for skill in l_target_skills:
					if skill == skill_to:
						replaced_l_target_skills.append(skill_from)
					else:
						replaced_l_target_skills.append(skill)
				optimal_source_skills[target_skills][req] = replaced_l_target_skills.copy()


with open("/home/jsw7460/mnt/comde_datasets/source_skills/kitchen/optimal_three_skills", "wb") as f:
	pickle.dump(optimal_source_skills, f)