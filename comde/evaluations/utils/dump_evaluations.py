from typing import Union
from pathlib import Path


def dump_eval_logs(save_path: Union[str, Path], eval_str: str):
	print(eval_str)
	print(f"Text file saved to {save_path}")

	with open(save_path, "a") as f:
		f.write(eval_str)
