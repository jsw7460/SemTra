from typing import List, Dict, Union, Optional

import jax
import numpy as np
from transformers import AutoTokenizer, FlaxCLIPTextModel


tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # type: Optional[AutoTokenizer,]
model = None  # type: Optional[FlaxBertModel]


@jax.jit
def _clip_encode(**kwargs):
	return model(**kwargs)["last_hidden_state"]


def clip_forward(languages: Union[str, List[str]]) -> Dict[str, Union[np.ndarray, Dict]]:
	if model is None:
		_init_pretrained_model()
	encoded_input = tokenizer(languages, return_tensors='np', padding=True)

	for k, v in encoded_input.items():
		encoded_input[k] = v[:, :77]
	language_embedding = _clip_encode(**encoded_input)

	output_dict = {
		**encoded_input,
		"language_embedding": language_embedding
	}

	return output_dict


def _init_pretrained_model():
	global model
	model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
