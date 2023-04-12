import pickle
import numpy as np
from itertools import product

from pathlib import Path

from transformers import BertTokenizer, BertModel
import torch as th

ONEHOT_SKILLS_MAPPING = {
		"box": 0,
		"puck": 1,
		"handle": 2,
		"drawer": 3,
		"button": 4,
		"lever": 5,
		"door": 6,
		"stick": 7
	}

SKILL_INDEX_MAPPING = {str(v): k for k, v in ONEHOT_SKILLS_MAPPING.items()}

if __name__ == "__main__":

	# === Hyper parameters
	save_dir = "/home/jsw7460/mnt/comde_datasets/language_embeddings/bert_mappings/language_guidance/"
	topic = "wind_seq_rep_rev||windonly"

	main_texts = []
	compositions = [
		"sequential", "reverse", "replace 3 with 1", "replace 6 with 1", "replace 4 with 1",
		"replace 6 with 3", "replace 4 with 3", "replace 1 with 6", "replace 1 with 3", "replace 3 with 4",
		"replace 3 with 6", "replace 6 with 4", "replace 1 with 4", "replace 4 with 6",
	]
	wind_scale = [-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

	for intent, comp in product(wind_scale, compositions):
		text = f"x_wind_{intent} || {comp}"
		main_texts.append(text)

	# text_variations: Key: main_texts // value: variations of main_texts
	texts_variations = {k: [] for k in main_texts}

	for k, v in texts_variations.items():
		chunk = k.split("wind_")[1]

		wind = chunk.split("||")[0]
		comp = chunk.split("||")[1]	# type: str

		for sk_idx in ["1", "3", "4", "6"]:
			comp = comp.replace(sk_idx, SKILL_INDEX_MAPPING[sk_idx])
		comp = comp.lstrip()
		comp = comp[0].upper() + comp[1:]
		# comp[0] = comp[0].upper()

		if wind[0] == "-":
			wind = wind.strip("-")
			leeside = "east"
			windside = "west"
		else:
			leeside = "west"
			windside = "east"

		# print(f"Wind: {wind}, leeside: {leeside}, windside: {windside}")
		texts_variations[k] = [f"{comp}. But, note that wind blows from {leeside} to {windside} by {wind}m/s."]

	assert main_texts == list(texts_variations.keys())

	# === Hyper parameters
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	model = BertModel.from_pretrained("bert-large-uncased").cuda()

	save_dict = {main_text: dict() for main_text in main_texts}

	time = 0
	with th.no_grad():
		for key, variations in texts_variations.items():
			for text in variations:
				time += 1

				chunk = key.split("||")[0]
				wind = chunk.split("wind_")[1]
				wind = eval(wind)
				sentence_vector = np.array([wind] * 1024)

				# encoded_input = tokenizer(text, return_tensors='pt').to("cuda:0")
				# output = model(**encoded_input)["last_hidden_state"]
				# sentence_vector = th.mean(output, dim=1).squeeze()
				# sentence_vector = sentence_vector.cpu().numpy()
				save_dict[key][text] = sentence_vector

	with open(Path(save_dir) / Path(f"{topic}_bert_mapping"), "wb") as f:
		pickle.dump(save_dict, f)
