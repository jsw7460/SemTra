from collections import defaultdict
from itertools import permutations

from comde.utils.common.natural_languages.lang_representation import SkillRepresentation

main_texts = [
		"stop",
		"straight",
		"left",
		"right",
	]
texts_variations = {
	"stop": ["stop"],
	"straight": ["go straight"],
	"left": ["turn left"],
	"right": ["turn right"],
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

SEQUENTIAL_REQUIREMENTS = [
	"sequential",
	"reverse"
]

for (a, b) in list(permutations(range(4), 2)):
	SEQUENTIAL_REQUIREMENTS.append(f"replace {a} with {b}")

SEQUENTIAL_REQUIREMENTS_VARIATIONS = {
	"sequential": [
    "Follow this driving video step by step.",
    "Go through this driving video in sequential order.",
    "Follow the instructions in this video in the given order.",
    "Mimic the actions shown in this driving video in a sequential manner.",
    "Emulate the steps demonstrated in this video for driving.",
    "Adhere to the sequence of actions presented in this driving video.",
    "Follow along with this video to drive following the given order.",
    "Replicate the driving techniques shown in this video in the correct order.",
    "Mirror the actions demonstrated in this video to drive accordingly.",
    "Navigate the road by following the instructions provided in this video."
],
	"reverse": [
    "Follow this driving video in reverse.",
    "Go through this driving video in the opposite order.",
    "Follow the instructions in this video in reverse order.",
    "Mimic the actions shown in this driving video in reverse.",
    "Emulate the steps demonstrated in this video for driving but in reverse order.",
    "Adhere to the sequence of actions presented in this driving video, but in reverse.",
    "Follow along with this video to drive in the opposite direction as shown.",
    "Replicate the driving techniques shown in this video, but in reverse order.",
    "Mirror the actions demonstrated in this video to drive in the opposite manner.",
    "Navigate the road by following the instructions provided in this video, but in reverse order."
],
	"replace 0 with 1": [
    "Proceed straight instead of stopping.",
    "Keep going forward instead of coming to a halt.",
    "Continue in a straight line instead of pausing.",
    "Maintain a straight path instead of stopping.",
    "Carry on without stopping, going straight ahead.",
    "Go straight without halting or pausing.",
    "Push forward in a straight direction without stopping.",
    "Advance without interruption, going straight.",
    "Move straight ahead without coming to a stop.",
    "Stay on course and continue straight without stopping."
],
	"replace 0 with 2": [
    "Make a left turn instead of stopping.",
    "Proceed with a left turn instead of coming to a halt.",
    "Continue by taking a left turn instead of stopping.",
    "Instead of stopping, make a turn to the left.",
    "Carry on with a left turn instead of pausing.",
    "Go left without stopping or halting.",
    "Push forward and make a left turn instead of stopping.",
    "Advance by making a left turn without interruption.",
    "Move left without coming to a stop.",
    "Stay on course and make a left turn without stopping."
]
,
	"replace 0 with 3": [
    "Make a right turn instead of stopping.",
    "Proceed with a right turn instead of coming to a halt.",
    "Continue by taking a right turn instead of stopping.",
    "Instead of stopping, make a turn to the right.",
    "Carry on with a right turn instead of pausing.",
    "Go right without stopping or halting.",
    "Push forward and make a right turn instead of stopping.",
    "Advance by making a right turn without interruption.",
    "Move right without coming to a stop.",
    "Stay on course and make a right turn without stopping."
]
,
	"replace 1 with 0": [
    "Stop and come to a halt instead of continuing straight.",
    "Don't proceed straight, but stop and come to a complete stop.",
    "Rather than going straight, stop and come to a standstill.",
    "Instead of continuing forward, pause and come to a stop.",
    "Cease forward movement and come to a stop instead of going straight.",
    "Avoid going straight and choose to stop and come to a halt.",
    "Refrain from proceeding straight and opt to stop and come to a stop.",
    "Do not continue straight; instead, come to a stop and halt.",
    "Instead of moving straight ahead, stop and come to a complete stop.",
    "Choose to stop and come to a standstill instead of continuing straight."
],
	"replace 1 with 2": [
    "Make a left turn instead of going straight.",
    "Avoid going straight and turn left instead.",
    "Opt for a left turn instead of continuing straight.",
    "Choose to make a left turn instead of proceeding straight.",
    "Instead of going straight, take a left turn.",
    "Go left instead of continuing straight.",
    "Prefer making a left turn over going straight.",
    "Rather than proceeding straight, turn left.",
    "Make a left turn instead of continuing in a straight line.",
    "In lieu of going straight, make a left turn."
],
	"replace 1 with 3": [
    "Make a right turn instead of going straight.",
    "Avoid going straight and turn right instead.",
    "Opt for a right turn instead of continuing straight.",
    "Choose to make a right turn instead of proceeding straight.",
    "Instead of going straight, take a right turn.",
    "Go right instead of continuing straight.",
    "Prefer making a right turn over going straight.",
    "Rather than proceeding straight, turn right.",
    "Make a right turn instead of continuing in a straight line.",
    "In lieu of going straight, make a right turn."
],
	"replace 2 with 0": [
    "Stop and come to a halt instead of making a left turn.",
    "Rather than making a left turn, stop and come to a complete stop.",
    "Don't proceed with a left turn, but instead stop and come to a standstill.",
    "Instead of turning left, pause and come to a stop.",
    "Choose to come to a stop instead of making a left turn.",
    "Avoid making a left turn and opt to stop and come to a halt.",
    "Refrain from making a left turn and choose to stop and come to a stop.",
    "Do not turn left; instead, come to a stop and halt.",
    "Instead of making a left turn, come to a complete stop.",
    "Opt for stopping and coming to a standstill instead of making a left turn."
],
	"replace 2 with 1": [
    "Go straight ahead instead of making a left turn.",
    "Avoid making a left turn and continue straight ahead.",
    "Opt for proceeding straight instead of making a left turn.",
    "Choose to continue straight ahead instead of turning left.",
    "Instead of turning left, continue straight ahead.",
    "Proceed straight ahead without making a left turn.",
    "Prefer going straight ahead over making a left turn.",
    "Rather than turning left, continue in a straight line.",
    "Continue straight ahead instead of making a left turn.",
    "In lieu of making a left turn, go straight ahead."
],
	"replace 2 with 3": [
    "Make a right turn instead of making a left turn.",
    "Avoid making a left turn and opt for a right turn instead.",
    "Opt for a right turn instead of proceeding with a left turn.",
    "Choose to make a right turn instead of turning left.",
    "Instead of turning left, make a right turn.",
    "Go right instead of making a left turn.",
    "Prefer making a right turn over making a left turn.",
    "Rather than turning left, turn right.",
    "Make a right turn instead of turning left.",
    "In lieu of making a left turn, make a right turn."
],
	"replace 3 with 0": [
    "Stop and come to a halt instead of making a right turn.",
    "Rather than making a right turn, stop and come to a complete stop.",
    "Don't proceed with a right turn, but instead stop and come to a standstill.",
    "Instead of turning right, pause and come to a stop.",
    "Choose to come to a stop instead of making a right turn.",
    "Avoid making a right turn and opt to stop and come to a halt.",
    "Refrain from making a right turn and choose to stop and come to a stop.",
    "Do not turn right; instead, come to a stop and halt.",
    "Instead of making a right turn, come to a complete stop.",
    "Opt for stopping and come to a standstill instead of making a right turn."
],
	"replace 3 with 1": [
    "Continue straight ahead instead of making a right turn.",
    "Avoid making a right turn and continue straight ahead instead.",
    "Opt for proceeding straight ahead instead of making a right turn.",
    "Choose to continue straight ahead instead of turning right.",
    "Instead of turning right, continue straight ahead.",
    "Proceed straight ahead without making a right turn.",
    "Prefer going straight ahead over making a right turn.",
    "Rather than turning right, continue in a straight line.",
    "Continue straight ahead instead of making a right turn.",
    "In lieu of making a right turn, go straight ahead."
],
	"replace 3 with 2": [
    "Make a left turn instead of making a right turn.",
    "Avoid making a right turn and opt for a left turn instead.",
    "Opt for a left turn instead of proceeding with a right turn.",
    "Choose to make a left turn instead of turning right.",
    "Instead of turning right, make a left turn.",
    "Go left instead of making a right turn.",
    "Prefer making a left turn over making a right turn.",
    "Rather than turning right, turn left.",
    "Make a left turn instead of turning right.",
    "In lieu of making a right turn, make a left turn."
]
}


NON_FUNCTIONALITIES_VARIATIONS = {"vehicle": ["type of vehicle"]}