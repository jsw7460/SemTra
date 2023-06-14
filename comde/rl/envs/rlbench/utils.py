import random
from collections import defaultdict
from itertools import permutations
from itertools import product
from typing import List

from comde.utils.common.natural_languages.lang_representation import SkillRepresentation
from comde_rlbench.RLBench.rlbench import comde_tasks
from comde_rlbench.RLBench.rlbench.comde_const import COMDE_SKILLS_TO_IDX
from comde_rlbench.RLBench.rlbench.comde_const import COMDE_WEIGHTS
from rlbench.tasks.close_door import CloseDoor
from rlbench.tasks.close_drawer import CloseDrawer
from rlbench.tasks.close_fridge import CloseFridge
from rlbench.tasks.close_microwave import CloseMicrowave
from rlbench.tasks.lamp_off import LampOff
from rlbench.tasks.lamp_on import LampOn
from rlbench.tasks.open_box import OpenBox
from rlbench.tasks.open_door import OpenDoor
from rlbench.tasks.open_drawer import OpenDrawer
from rlbench.tasks.press_switch import PressSwitch
from rlbench.tasks.push_button import PushButton
from rlbench.tasks.slide_block_to_target import SlideBlockToTarget

COMDE_IDX_TO_SKILLS = {v: k for k, v in COMDE_SKILLS_TO_IDX.items()}
COMDE_SKILLS_POSITIONS = {
	"leftmost": [0, 1, 2],
	"left": [3, 4, 5],
	"right": [6, 7, 8],
	"rightmost": [9, 10, 11]
}


def tasks_to_idxs(tasks: List) -> List[int]:
	return [COMDE_SKILLS_TO_IDX[task] for task in tasks]  # [1, 3, 7, 9]


COMDE_TASKS_INDICES = {
	"1": tasks_to_idxs([OpenDoor, OpenDrawer, PushButton, CloseMicrowave]),
	"2": tasks_to_idxs([OpenDoor, OpenDrawer, PushButton, OpenBox]),
	"3": tasks_to_idxs([OpenDoor, OpenDrawer, PushButton, SlideBlockToTarget]),
	"4": tasks_to_idxs([OpenDoor, OpenDrawer, LampOn, CloseMicrowave]),
	"5": tasks_to_idxs([OpenDoor, OpenDrawer, LampOn, OpenBox]),
	"6": tasks_to_idxs([OpenDoor, OpenDrawer, LampOn, SlideBlockToTarget]),
	"7": tasks_to_idxs([OpenDoor, OpenDrawer, PressSwitch, CloseMicrowave]),
	"8": tasks_to_idxs([OpenDoor, OpenDrawer, PressSwitch, OpenBox]),
	"9": tasks_to_idxs([OpenDoor, OpenDrawer, PressSwitch, SlideBlockToTarget]),
	"10": tasks_to_idxs([OpenDoor, CloseDrawer, PushButton, CloseMicrowave]),
	"11": tasks_to_idxs([OpenDoor, CloseDrawer, PushButton, OpenBox]),
	"12": tasks_to_idxs([OpenDoor, CloseDrawer, PushButton, SlideBlockToTarget]),
	"13": tasks_to_idxs([OpenDoor, CloseDrawer, LampOn, CloseMicrowave]),
	"14": tasks_to_idxs([OpenDoor, CloseDrawer, LampOn, OpenBox]),
	"15": tasks_to_idxs([OpenDoor, CloseDrawer, LampOn, SlideBlockToTarget]),
	"16": tasks_to_idxs([OpenDoor, CloseDrawer, PressSwitch, CloseMicrowave]),
	"17": tasks_to_idxs([OpenDoor, CloseDrawer, PressSwitch, OpenBox]),
	"18": tasks_to_idxs([OpenDoor, CloseDrawer, PressSwitch, SlideBlockToTarget]),
	"19": tasks_to_idxs([OpenDoor, LampOff, PushButton, CloseMicrowave]),
	"20": tasks_to_idxs([OpenDoor, LampOff, PushButton, OpenBox]),
	"21": tasks_to_idxs([OpenDoor, LampOff, PushButton, SlideBlockToTarget]),
	"22": tasks_to_idxs([OpenDoor, LampOff, LampOn, CloseMicrowave]),
	"23": tasks_to_idxs([OpenDoor, LampOff, LampOn, OpenBox]),
	"24": tasks_to_idxs([OpenDoor, LampOff, LampOn, SlideBlockToTarget]),
	"25": tasks_to_idxs([OpenDoor, LampOff, PressSwitch, CloseMicrowave]),
	"26": tasks_to_idxs([OpenDoor, LampOff, PressSwitch, OpenBox]),
	"27": tasks_to_idxs([OpenDoor, LampOff, PressSwitch, SlideBlockToTarget]),
	"28": tasks_to_idxs([CloseDoor, OpenDrawer, PushButton, CloseMicrowave]),
	"29": tasks_to_idxs([CloseDoor, OpenDrawer, PushButton, OpenBox]),
	"30": tasks_to_idxs([CloseDoor, OpenDrawer, PushButton, SlideBlockToTarget]),
	"31": tasks_to_idxs([CloseDoor, OpenDrawer, LampOn, CloseMicrowave]),
	"32": tasks_to_idxs([CloseDoor, OpenDrawer, LampOn, OpenBox]),
	"33": tasks_to_idxs([CloseDoor, OpenDrawer, LampOn, SlideBlockToTarget]),
	"34": tasks_to_idxs([CloseDoor, OpenDrawer, PressSwitch, CloseMicrowave]),
	"35": tasks_to_idxs([CloseDoor, OpenDrawer, PressSwitch, OpenBox]),
	"36": tasks_to_idxs([CloseDoor, OpenDrawer, PressSwitch, SlideBlockToTarget]),
	"37": tasks_to_idxs([CloseDoor, CloseDrawer, PushButton, CloseMicrowave]),
	"38": tasks_to_idxs([CloseDoor, CloseDrawer, PushButton, OpenBox]),
	"39": tasks_to_idxs([CloseDoor, CloseDrawer, PushButton, SlideBlockToTarget]),
	"40": tasks_to_idxs([CloseDoor, CloseDrawer, LampOn, CloseMicrowave]),
	"41": tasks_to_idxs([CloseDoor, CloseDrawer, LampOn, OpenBox]),
	"42": tasks_to_idxs([CloseDoor, CloseDrawer, LampOn, SlideBlockToTarget]),
	"43": tasks_to_idxs([CloseDoor, CloseDrawer, PressSwitch, CloseMicrowave]),
	"44": tasks_to_idxs([CloseDoor, CloseDrawer, PressSwitch, OpenBox]),
	"45": tasks_to_idxs([CloseDoor, CloseDrawer, PressSwitch, SlideBlockToTarget]),
	"46": tasks_to_idxs([CloseDoor, LampOff, PushButton, CloseMicrowave]),
	"47": tasks_to_idxs([CloseDoor, LampOff, PushButton, OpenBox]),
	"48": tasks_to_idxs([CloseDoor, LampOff, PushButton, SlideBlockToTarget]),
	"49": tasks_to_idxs([CloseDoor, LampOff, LampOn, CloseMicrowave]),
	"50": tasks_to_idxs([CloseDoor, LampOff, LampOn, OpenBox]),
	"51": tasks_to_idxs([CloseDoor, LampOff, LampOn, SlideBlockToTarget]),
	"52": tasks_to_idxs([CloseDoor, LampOff, PressSwitch, CloseMicrowave]),
	"53": tasks_to_idxs([CloseDoor, LampOff, PressSwitch, OpenBox]),
	"54": tasks_to_idxs([CloseDoor, LampOff, PressSwitch, SlideBlockToTarget]),
	"55": tasks_to_idxs([CloseFridge, OpenDrawer, PushButton, CloseMicrowave]),
	"56": tasks_to_idxs([CloseFridge, OpenDrawer, PushButton, OpenBox]),
	"57": tasks_to_idxs([CloseFridge, OpenDrawer, PushButton, SlideBlockToTarget]),
	"58": tasks_to_idxs([CloseFridge, OpenDrawer, LampOn, CloseMicrowave]),
	"59": tasks_to_idxs([CloseFridge, OpenDrawer, LampOn, OpenBox]),
	"60": tasks_to_idxs([CloseFridge, OpenDrawer, LampOn, SlideBlockToTarget]),
	"61": tasks_to_idxs([CloseFridge, OpenDrawer, PressSwitch, CloseMicrowave]),
	"62": tasks_to_idxs([CloseFridge, OpenDrawer, PressSwitch, OpenBox]),
	"63": tasks_to_idxs([CloseFridge, OpenDrawer, PressSwitch, SlideBlockToTarget]),
	"64": tasks_to_idxs([CloseFridge, CloseDrawer, PushButton, CloseMicrowave]),
	"65": tasks_to_idxs([CloseFridge, CloseDrawer, PushButton, OpenBox]),
	"66": tasks_to_idxs([CloseFridge, CloseDrawer, PushButton, SlideBlockToTarget]),
	"67": tasks_to_idxs([CloseFridge, CloseDrawer, LampOn, CloseMicrowave]),
	"68": tasks_to_idxs([CloseFridge, CloseDrawer, LampOn, OpenBox]),
	"69": tasks_to_idxs([CloseFridge, CloseDrawer, LampOn, SlideBlockToTarget]),
	"70": tasks_to_idxs([CloseFridge, CloseDrawer, PressSwitch, CloseMicrowave]),
	"71": tasks_to_idxs([CloseFridge, CloseDrawer, PressSwitch, OpenBox]),
	"72": tasks_to_idxs([CloseFridge, CloseDrawer, PressSwitch, SlideBlockToTarget]),
	"73": tasks_to_idxs([CloseFridge, LampOff, PushButton, CloseMicrowave]),
	"74": tasks_to_idxs([CloseFridge, LampOff, PushButton, OpenBox]),
	"75": tasks_to_idxs([CloseFridge, LampOff, PushButton, SlideBlockToTarget]),
	"76": tasks_to_idxs([CloseFridge, LampOff, LampOn, CloseMicrowave]),
	"77": tasks_to_idxs([CloseFridge, LampOff, LampOn, OpenBox]),
	"78": tasks_to_idxs([CloseFridge, LampOff, LampOn, SlideBlockToTarget]),
	"79": tasks_to_idxs([CloseFridge, LampOff, PressSwitch, CloseMicrowave]),
	"80": tasks_to_idxs([CloseFridge, LampOff, PressSwitch, OpenBox]),
	"81": tasks_to_idxs([CloseFridge, LampOff, PressSwitch, SlideBlockToTarget]),
}

RLBENCH_ALL_TASKS = list(product(*list(COMDE_SKILLS_POSITIONS.values())))

SEQUENTIAL_REQUIREMENT = ["sequential", "reverse"]
for (a, b) in list(permutations(COMDE_WEIGHTS.keys(), 2)):
	SEQUENTIAL_REQUIREMENT.append(f"replace {a} with {b}")

WEIGHT_TO_ADJECTIVE = {}
for skill, weights in COMDE_WEIGHTS.items():
	WEIGHT_TO_ADJECTIVE[skill] = {weight: adj for weight, adj in zip(weights, ["normal", "heavy", "light"])}


def get_task_class(task: List[int]):
	candidates = []
	for k, v in COMDE_TASKS_INDICES.items():
		if set(task).issubset(set(v)):
			candidates.append(k)
	task_idx = random.choice(candidates)
	task_class = getattr(comde_tasks, f"ComdeTask{task_idx}", None)

	return task_class


def get_weight(skill_idx: int, adj: str):
	"""
		adj: default, heavy, light
	"""
	obj_weights = COMDE_WEIGHTS[skill_idx]
	if adj == "default":
		return obj_weights[0]
	elif adj == "heavy":
		return obj_weights[1]
	elif adj == "light":
		return obj_weights[2]
	else:
		raise NotImplementedError(f"Undefined weight adjective: {adj}")


def object_in_task(task: str):
	if "door" in task:
		return "door"
	elif "fridge" in task:
		return "fridge"
	elif "drawer" in task:
		return "drawer"
	elif "lamp" in task:
		return "lamp"
	elif "button" in task:
		return "button"
	elif "switch" in task:
		return "switch"
	elif "microwave" in task:
		return "microwave"
	elif "box" in task:
		return "box"
	elif "block" in task:
		return "block"
	else:
		raise NotImplementedError(f"{task} has no object identifier")


# NOTE: """order matters""". it mapped to one-hot representation.
main_texts = [
		"open door",
		"close door",
		"close fridge",
		"open drawer",
		"close drawer",
		"lamp off",
		"push button",
		"lamp on",
		"press switch",
		"close microwave",
		"open box",
		"slide block to target",
	]
texts_variations = {
	mt: [mt] for mt in main_texts
}

skill_infos = defaultdict(list)
for idx, (key, variations) in enumerate(texts_variations.items()):
	for text in variations:
		skill_rep = SkillRepresentation(
			title=key,
			variation=text,
			vec="override this using language model",
			index=idx
		)
		skill_infos[key].append(skill_rep)
