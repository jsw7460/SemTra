from collections import defaultdict

import torch as th
from transformers import BertTokenizer, BertModel

from comde.utils.common.lang_representation import SkillRepresentation


if __name__ == "__main__":

	# === Hyper parameters
	save_dir = "/home/jsw7460/mnt/comde_datasets/language_embeddings/bert_mappings/language_guidance/"
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
	texts_variations = {
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
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	model = BertModel.from_pretrained("bert-large-uncased").cuda()

	save_dict = defaultdict(list)

	with th.no_grad():
		for idx, (key, variations) in enumerate(texts_variations.items()):
			for text in variations:
				encoded_input = tokenizer(text, return_tensors='pt').to("cuda:0")
				output = model(**encoded_input)["last_hidden_state"]
				sentence_vector = th.mean(output, dim=1).squeeze()
				sentence_vector = sentence_vector.cpu().numpy()

				SkillRepresentation(
					title=key,
					variation=text,
					vec=sentence_vector,
					index=idx
				)

				save_dict[key].append(sentence_vector)

	print(save_dict["box"][0])

	# save_dict = defaultdict(list)
	#
	# for idx, title in enumerate(main_texts):
	# 	# print(idx, title, variation, tensor.mean())
	# 	vectors_variations = text_features[title]
	#
	# 	for variation, tensor in vectors_variations.items():
	# 		skill_representation = SkillRepresentation(
	# 			title=title,
	# 			variation=variation,
	# 			vec=np.squeeze(tensor.numpy(), axis=0),
	# 			index=idx
	# 		)
	# 		save_dict[title].append(skill_representation)

	# with open("/home/jsw7460/mnt/comde_datasets/clip_mappings/metaworld/clip_mapping", "wb") as f:
	# 	pickle.dump(save_dict, f)
