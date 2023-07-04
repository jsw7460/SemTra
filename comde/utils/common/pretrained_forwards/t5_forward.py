from typing import List, Dict, Union, Optional

import jax
import numpy as np
from transformers import AutoTokenizer
from transformers.models.t5.modeling_flax_t5 import FlaxT5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("t5-base") # type: Optional[AutoTokenizer,]
model = None  # type: Optional[FlaxBertModel]


@jax.jit
def _t5_encode(**kwargs):
	return model(**kwargs)["last_hidden_state"]


def t5_forward(languages: Union[str, List[str]]) -> Dict[str, Union[np.ndarray, Dict]]:
	if model is None:
		_init_pretrained_model()

	encoded_input = tokenizer(languages, return_tensors='np', padding=True)
	language_embedding = _t5_encode(**encoded_input)

	output_dict = {
		**encoded_input,
		"language_embedding": language_embedding
	}

	return output_dict


def _init_pretrained_model():
	global model
	model = FlaxT5EncoderModel.from_pretrained("t5-base")
