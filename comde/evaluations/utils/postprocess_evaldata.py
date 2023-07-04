from typing import Dict

import numpy as np
from math import fabs


def postprocess_eval_data(eval_infos: Dict, n_tasks: int):
	n_envs = len(eval_infos)
	returns = []
	for k in range(len(eval_infos)):
		rewards = eval_infos[f"env_{k}"]["rewards"]
		rewards = np.array(rewards)

		last_success = 0
		for t, rew in enumerate(rewards):
			if rew > 0:
				if fabs(last_success - t) <= 2:
					rewards[last_success: t + 1] = 0
				if rew > 1:
					rewards[t - 1] = 0
					rewards[t] = 0
				last_success = t

		_return = np.sum(rewards)
		eval_infos[f"env_{k}"]["return"] = _return
		returns.append(_return)

	returns = np.array(returns)

	eval_fmt = f"Returns: {returns} \n" \
			   f"Total sum: {returns.sum()} among {n_envs} tasks \n" \
			   f"Total success ratio: {100 * (returns.sum() / n_tasks)}"

	return eval_infos, eval_fmt
