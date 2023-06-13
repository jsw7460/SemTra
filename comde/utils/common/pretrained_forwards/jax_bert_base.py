from typing import List, Dict, Union

import jax
import numpy as np
from transformers import FlaxBertModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("bert-base-uncased")


@jax.jit
def _bert_encode(**kwargs):
	return model(**kwargs)["last_hidden_state"]


def bert_base_forward(languages: Union[str, List[str]]) -> Dict[str, Union[np.ndarray, Dict]]:
	encoded_input = tokenizer(languages, return_tensors='np', padding=True)
	language_embedding = _bert_encode(**encoded_input)

	output_dict = {
		**encoded_input,
		"language_embedding": language_embedding
	}

	return output_dict
