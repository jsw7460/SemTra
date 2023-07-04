import matplotlib.pyplot as plt
import numpy as np


def split_string(string, max_len=50):
	lines = []
	current_line = ""

	# Split the string into words
	words = string.split()

	for word in words:
		if len(current_line + word) <= max_len:
			current_line += word + " "
		else:
			lines.append(current_line.strip())
			current_line = word + " "

	# Add the last line
	lines.append(current_line.strip())

	return "\n".join(lines)


def dump_attention_weights_images(
	path: str,
	natural_language: str,
	attention_matrix: np.ndarray
):
	fig, ax = plt.subplots()
	heatmap = ax.imshow(attention_matrix, cmap='hot', interpolation='nearest')

	# Add colorbar
	plt.colorbar(heatmap)

	# Set x and y axis labels

	ax.set_xlabel(split_string(natural_language))
	ax.set_ylabel('Skills')

	# Show the plot
	plt.savefig(fname=path)
	plt.close()
