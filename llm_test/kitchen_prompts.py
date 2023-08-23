import pickle
import random

random.seed(777)
kitchen_system_prompt = """ You are a model trained to sequence tasks. You have a list of seven skills:

1. turn on bottom burner
2. turn on top burner
3. turn on light switch
4. slide cabinet
5. open hinge cabinet
6. open microwave
7. move kettle

Your task is to order these skills in a logical sequence to solve the instruction.
After all skills have been used, please write "Finish".
Please list the skills in the order they should be performed, such as "1. First skill 2. Second skill", and so on.
Do not include any other words in your response and your answer may not contain all skills (for instance, there might be 3~4 skills).

"""

onehot_skills_mapping = {
	'bottom burner': 0,
	'top burner': 1,
	'light switch': 2,
	'slide cabinet': 3,
	'hinge cabinet': 4,
	'microwave': 5,
	'kettle': 6,
}

skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

skill_verb_mapping = {
	'bottom burner': "turn on bottom burner",
	'top burner': "turn on top burner",
	'light switch': "turn on light switch",
	'slide cabinet': "slide cabinet",
	'hinge cabinet': "open hinge cabinet",
	'microwave': "open microwave",
	'kettle': "move kettle",
}

verb_skill_mapping = {v: k for k, v in skill_verb_mapping.items()}

with open("/home/jsw7460/mnt/comde_datasets/source_skills/kitchen/optimal_four_skills", "rb") as f:
	data = pickle.load(f)


def replace_idx_so_skill(sentence: str) -> str:
	sentence = sentence.split()
	for t, word in enumerate(sentence):
		if word.isdigit():
			sentence[t] = skill_verb_mapping[skill_index_mapping[int(word)]]
	return " ".join(sentence)


def replace_template(task, sequential_requirement, wind_type):
	user_comment = f"""
The instruction is:
Do {task}, but {sequential_requirement} when the {wind_type} wind blows. 
"""
	return user_comment

def sequential_template(task, sequential_requirement, wind_type):
	user_comment = f"""
The instruction is:
Do {task} {sequential_requirement} when the {wind_type} wind blows. """
	return user_comment


kitchen_description = []
sequential_description = []
reverse_description = []
replace_description = []

for _answer, source in data.items():
	answer = _answer.split(" ")
	answer = [a.replace("_", " ") for a in answer]
	for seq_req, source_skills in source.items():
		if source_skills is not None:
			# print(source_skills, seq_req, answer)
			v_source_skills = [skill_verb_mapping[sk] for sk in source_skills]
			v_answer = [skill_verb_mapping[sk] for sk in answer]
			if "replace" in seq_req:
				seq_req = replace_idx_so_skill(seq_req)
			task_str = ", ".join(v_source_skills[: -1])
			task_str = task_str + ", and then " + v_source_skills[-1]
			for wind_type in ["breeze", "flurry", "gust"]:
				if "replace" in seq_req:
					command = replace_template(task_str, seq_req, wind_type)
					replace_description.append((command, v_answer))
				elif "sequential" in seq_req:
					command = sequential_template(task_str, "in sequential order", wind_type)
					sequential_description.append((command, v_answer))
				elif "reverse" in seq_req:
					command = sequential_template(task_str, "but in reverse order", wind_type)
					reverse_description.append((command, v_answer))
				else:
					raise NotImplementedError(f"?????: {seq_req}")

random.shuffle(sequential_description)
random.shuffle(reverse_description)
random.shuffle(replace_description)

kitchen_description = sequential_description[: 8] + reverse_description[: 8] + replace_description[: 8]
# random.shuffle(kitchen_description)

