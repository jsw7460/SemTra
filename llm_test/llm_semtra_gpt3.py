import openai

openai.api_key = ""
ENGINE = "text-davinci-002"

import argparse
import os

from kitchen_prompts import (
	kitchen_description,
	kitchen_system_prompt
)

from metaworld_prompts import (
	metaworld_description,
	metaworld_system_prompt
)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='experiments settings')
	parser.add_argument('--type', type=str, default='kitchen')
	args = parser.parse_args()

	if args.type == 'kitchen':
		instructions = kitchen_description
		system_prompt = kitchen_system_prompt

	elif args.type == "metaworld":
		instructions = metaworld_description
		system_prompt = metaworld_system_prompt

	if not os.path.exists("./llm_results/{}".format(args.type)):
		os.mkdir("./llm_results/{}".format(args.type))

	for idx, description_and_solution in enumerate(instructions):
		case_str = "Case {}: ".format(idx)
		termination_string = "Finish"
		description = description_and_solution[0]
		solution = description_and_solution[1]
		do_skills = []

		response = openai.Completion.create(
			model=ENGINE,
			prompt=system_prompt + description,
			temperature=0,
			max_tokens=256
		)

		response_message = response.choices[0].text
		print(response_message)
		with open("./llm_results/{}/semtra_gpt3_{}.txt".format(args.type, idx), 'w') as f:
			f.write(description + response_message + "\n\n" + "Solution: \n" + "\n".join(solution))
		# break
