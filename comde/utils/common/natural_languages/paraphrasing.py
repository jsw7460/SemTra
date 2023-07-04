from typing import NamedTuple

import numpy as np
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase").to("cuda")


class SequentialRequirement(NamedTuple):
	title_natural_language: str
	variation_natural_language: str
	title_vec: np.ndarray
	variation_vec: np.ndarray


class NonFunctionality(NamedTuple):
	natural_language: str  # apply wind on close the box
	non_functionality: str  # wind
	skill: str  # close the box
	natural_language_vec: np.ndarray


class Parameter(NamedTuple):
	natural_language: str
	vec: np.ndarray = None


class Template(NamedTuple):
	sequential_requirement: SequentialRequirement
	non_functionality: NonFunctionality
	parameter: Parameter


def get_paraphrased_sentences(sentence, num_return_sequences=5, num_beams=5, replacements = None):
	# tokenize the text to be form of a list of token IDs
	inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt").to("cuda")
	# generate the paraphrased sentences
	outputs = model.generate(
		**inputs,
		num_beams=num_beams,
		num_return_sequences=num_return_sequences,
	)
	# decode the generated sentences using the tokenizer to get them back to text
	paraphrased = tokenizer.batch_decode(outputs, skip_special_tokens=True)

	if replacements is not None:
		for j in range(len(paraphrased)):
			replaced_sentence = paraphrased[j]
			for (from_, to_) in replacements:
				replaced_sentence = replaced_sentence.replace(str(from_), str(to_))

			paraphrased[j] = replaced_sentence
	return paraphrased
