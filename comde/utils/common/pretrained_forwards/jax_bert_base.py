from typing import List, Dict

import jax
import numpy as np
from transformers import BertTokenizer, FlaxBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertModel.from_pretrained("bert-base-uncased")


@jax.jit
def _bert_encode(**kwargs):
	return model(**kwargs)["last_hidden_state"]


def bert_base_forward(languages: List[str]) -> Dict[str, np.ndarray]:
	encoded_input = tokenizer(languages, return_tensors='np', padding=True)
	language_embedding = _bert_encode(**encoded_input)

	output_dict = encoded_input
	output_dict["language_embedding"] = language_embedding

	return output_dict
