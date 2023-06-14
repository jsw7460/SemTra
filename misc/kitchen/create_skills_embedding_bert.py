import sys
sys.path.append("/home/jsw7460/comde")
import pickle
from collections import defaultdict

import torch as th
from comde.utils.common.natural_languages.lang_representation import SkillRepresentation

if __name__ == "__main__":

	# === Hyper parameters
	save_dir = "/home/jsw7460/mnt/comde_datasets/language_embeddings/bert_mappings/language_guidance/"
	# NOTE: """order matters""". it mapped to one-hot representation.
	main_texts = [
		"bottom burner",
		"top burner",
		"light switch",
		"slide cabinet",
		"hinge cabinet",
		"microwave",
		"kettle",
	]
	texts_variations = {
		"bottom burner": ["bottom burner"],
		"top burner": ["top burner"],
		"light switch": ["light switch"],
		"slide cabinet": ["slide cabinet"],
		"hinge cabinet": ["hinge cabinet"],
		"microwave": ["microwave"],
		"kettle": ["kettle"],
	}
	# === Hyper parameters

	save_dict = defaultdict(list)

	with th.no_grad():
		for idx, (key, variations) in enumerate(texts_variations.items()):
			for text in variations:

				skill_rep = SkillRepresentation(
					title=key,
					variation=text,
					vec="override this using language model",
					index=idx
				)

				save_dict[key].append(skill_rep)

	with open("/home/jsw7460/mnt/comde_datasets/language_embeddings/bert_mappings/kitchen/bert_base_skills_mapping",
			  "wb") as f:
		pickle.dump(save_dict, f)
