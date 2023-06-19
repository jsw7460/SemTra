import pickle
from collections import defaultdict

import clip
import numpy as np
import torch

from comde.utils.common.natural_languages.lang_representation import SkillRepresentation

AVAILABLE_MODELS = ['RN50',
					'RN101',
					'RN50x4',
					'RN50x16',
					'RN50x64',
					'ViT-B/32',
					'ViT-B/16',
					'ViT-L/14',
					'ViT-L/14@336px']

if __name__ == "__main__":

	# === Hyper parameters
	save_dir = "/home/jsw7460/mnt/comde_datasets/clip_mappings/language_guidance/"
	# NOTE: """order matters""". it mapped to one-hot representation.
	main_texts = [
		"box",
		"puck",
		"handle",
		"drawer",
		"button",
		"lever",
		"door",
		"stick"
	]
	variations = {
		"box": ["close box"],
		"puck": ["slide puck"],
		"handle": ["pull handle"],
		"drawer": ["close drawer"],
		"button": ["press button"],
		"lever": ["pull lever"],
		"door": ["open door"],
		"stick": ["insert stick"]
	}
	# === Hyper parameters

	# text_tokens = [clip.tokenize(v) for v in variations]

	model_name = "ViT-B/32"
	assert model_name in AVAILABLE_MODELS, "See above avilable models"
	model, preprocess = clip.load(model_name)
	model.eval()

	text_features = defaultdict(dict)

	for skill in main_texts:
		for variation in variations[skill]:
			with torch.no_grad():
				text_features[skill][variation] = model.encode_text(clip.tokenize(variation)).float()

	save_dict = defaultdict(list)

	for idx, title in enumerate(main_texts):
		# print(idx, title, variation, tensor.mean())
		vectors_variations = text_features[title]

		for variation, tensor in vectors_variations.items():
			skill_representation = SkillRepresentation(
				title=title,
				variation=variation,
				vec=np.squeeze(tensor.numpy(), axis=0),
				index=idx
			)
			save_dict[title].append(skill_representation)

	with open("/home/jsw7460/mnt/comde_datasets/clip_mappings/metaworld/clip_mapping", "wb") as f:
		pickle.dump(save_dict, f)
