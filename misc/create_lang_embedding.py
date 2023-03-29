import pickle

from pathlib import Path

import clip
import torch

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
	topic = "speed"
	main_texts = ["moderate", "slow", "fast"]
	texts_variations = {
		"moderate": ["moderate"],
		"slow": ["slow"],
		"fast": ["fast"]
	}
	# === Hyper parameters

	text_tokens = {
		k: clip.tokenize(v).cuda() for k, v in texts_variations.items()
	}

	model_name = "ViT-B/32"
	assert model_name in AVAILABLE_MODELS, "See above avilable models"
	model, preprocess = clip.load(model_name)
	model.cuda().eval()

	with torch.no_grad():
		text_features = {
			k: model.encode_text(v).float() for k, v in text_tokens.items()
		}

	save_dict = {main_text: dict() for main_text in main_texts}
	for k, tensors in text_features.items():
		# k: moderate
		for i, tensor in enumerate(tensors):
			save_dict[k][texts_variations[k][i]] = tensor.cpu().numpy()

	# for i, text in enumerate(texts_variations):
	# 	save_dict[f"{main_texts}"][text] = text_features[i].cpu().numpy()

	with open(Path(save_dir) / Path(f"{topic}_clip_mapping_{model_name.replace('/', '_')}"), "wb") as f:
		pickle.dump(save_dict, f)
