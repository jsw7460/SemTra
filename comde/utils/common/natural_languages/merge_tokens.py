from collections import defaultdict
from typing import List, Dict, Union, Tuple

from comde.utils.common.natural_languages.lang_representation import SkillRepresentation
from comde.rl.envs.base import ComdeSkillEnv
from comde.rl.envs.utils.skill_to_vec import SkillInfoEnv

TokenDict = Dict[str, List[SkillRepresentation]]
OffsetInfo = Dict[str, int]


def merge_env_tokens(envs: List[Union[ComdeSkillEnv, SkillInfoEnv]]) -> Tuple[TokenDict, OffsetInfo]:

	# If skill translator is trained for multiple environments simultaneously, this function is necessary
	token_dicts = [env.skill_infos for env in envs]
	offset_info = dict()

	merged = defaultdict(list)
	index_offset = 0
	for env, token_dict in zip(envs, token_dicts):
		offset_info[str(env)] = index_offset
		for skill, representations in token_dict.items():
			for i in range(len(representations)):
				rep = representations[i]
				idx = rep.index
				offset_rep = rep._replace(index=idx + index_offset)
				merged[skill].append(offset_rep)
		index_offset += len(token_dict)

	return merged, offset_info
