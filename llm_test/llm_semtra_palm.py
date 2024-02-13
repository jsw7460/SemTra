PaLM_API = ""
import argparse
import os

import google.generativeai as palm

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

	if not os.path.exists("./llm_results/{}".format(args.type)):
		os.mkdir("./llm_results/{}".format(args.type))

	if args.type == 'kitchen':
		instructions = kitchen_description
		system_prompt = kitchen_system_prompt

	elif args.type == "metaworld":
		instructions = metaworld_description
		system_prompt = metaworld_system_prompt

	palm.configure(api_key=PaLM_API)
	evaluation_check = []
	num_success_case = 0
	models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
	model = models[0].name
	print(model)

	for idx, description_and_solution in enumerate(instructions):
		case_str = "Case {}: ".format(idx)
		termination_string = "Finish"
		description = description_and_solution[0]
		solution = description_and_solution[1]
		do_skills = []

		completion = palm.generate_text(
			model=model,
			prompt=system_prompt + description,
			temperature=0,
			max_output_tokens=800,
		)

		# print(system_prompt + description)
		print(completion.result)
		print("\n\n\n")

		with open("./llm_results/{}/semtra_palm_{}.txt".format(args.type, idx), 'w') as f:
			f.write(description + completion.result + "\n\n" + "Solution: \n" + "\n".join(solution))
