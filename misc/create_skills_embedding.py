import pickle

import numpy as np

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
	main_texts = [
		"close box",
		"slide puck",
		"close drawer",
		"pull handle",
		"press button",
		"pull lever",
		"open door",
		"insert stick"
	]
	# === Hyper parameters

	text_tokens = [clip.tokenize(v).cuda() for v in main_texts]

	model_name = "ViT-B/32"
	assert model_name in AVAILABLE_MODELS, "See above avilable models"
	model, preprocess = clip.load(model_name)
	model.cuda().eval()

	with torch.no_grad():
		text_features = [model.encode_text(v).float() for v in text_tokens]

	save_dict = dict()
	for k, tensor in zip(main_texts, text_features):
		save_dict[k] = np.squeeze(tensor.cpu().numpy(), axis=0)

	for k, v in save_dict.items():
		print(k, v.shape)

	with open("/home/jsw7460/mnt/comde_datasets/clip_mappings/metaworld/clip_mapping", "wb") as f:
		pickle.dump(save_dict, f)
