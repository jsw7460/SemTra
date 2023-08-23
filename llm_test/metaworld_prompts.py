import pickle
import random

random.seed(777)
metaworld_system_prompt = """ You are a model trained to sequence tasks. You have a list of seven skills:

1. open the box
2. slide puck
3. pull up the handle
4. close the drawer
5. push the button
6. pull up the lever
7. open the door
8. insert the stick

Your task is to order these skills in a logical sequence to solve the instruction.
After all skills have been used, please write "Finish".
Please list the skills in the order they should be performed, such as "1. First skill 2. Second skill", and so on.
Do not include any other words in your response and your answer may not contain all skills (for instance, there might be 3~4 skills).

"""

onehot_skills_mapping = {
		"box": 0, "puck": 1, "handle": 2, "drawer": 3, "button": 4, "lever": 5, "door": 6, "stick": 7
	}

skill_index_mapping = {v: k for k, v in onehot_skills_mapping.items()}

skill_verb_mapping = {
	'box': "open the box",
	'puck': "slide puck",
	'handle': "pull up the handle",
	'drawer': "close the drawer",
	'button': "push the button",
	'lever': "pull up the lever",
	'door': "open the door",
	'stick': "insert the stick"
}

verb_skill_mapping = {v: k for k, v in skill_verb_mapping.items()}

with open("/home/jsw7460/mnt/comde_datasets/source_skills/metaworld/optimal_three_skills", "rb") as f:
	data = pickle.load(f)


def replace_idx_so_skill(sentence: str) -> str:
	sentence = sentence.split()
	for t, word in enumerate(sentence):
		if word.isdigit():
			sentence[t] = skill_verb_mapping[skill_index_mapping[int(word)]]
	return " ".join(sentence)


def replace_template(task, sequential_requirement, pa_sk, speed_type):
	user_comment = f"""
		The instruction is:
		Do {task}, but {sequential_requirement}. And please {pa_sk} {speed_type}. 
	"""
	return user_comment

def sequential_template(task, sequential_requirement, pa_sk, speed_type):
	user_comment = f"""
		The instruction is:
		Do {task} {sequential_requirement}. And please {pa_sk} {speed_type}. """
	return user_comment


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
			for speed_type in ["normal", "slow", "fast"]:
				pa_skill = random.choice(v_source_skills)

				if "replace" in seq_req:
					command = replace_template(task_str, seq_req, pa_skill, speed_type)
					replace_description.append((command, v_answer))
				elif "sequential" in seq_req:
					command = sequential_template(task_str, "in sequential order", pa_skill, speed_type)
					sequential_description.append((command, v_answer))
				elif "reverse" in seq_req:
					command = sequential_template(task_str, "but in reverse order", pa_skill, speed_type)
					reverse_description.append((command, v_answer))
				else:
					raise NotImplementedError(f"?????: {seq_req}")

random.shuffle(sequential_description)
random.shuffle(reverse_description)
random.shuffle(replace_description)

metaworld_description = sequential_description[: 8] + reverse_description[: 8] + replace_description[: 8]

