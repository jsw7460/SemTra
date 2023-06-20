import random
from typing import Dict, List

import jax.random
import jax.random
import numpy as np
import optax

from comde.baselines.algos.updates.prompt_dt import promptdt_update
from comde.baselines.algos.forwards import promptdt_forward as forward
from comde.baselines.architectures.vima import VIMA as _VIMA
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class VIMA(BaseLowPolicy):
    """
    VLPromptDT: Video-Language prompt DT
    Input of PromptDT
        1. State-action-reward history (Like DT)
        2. First image of skill & Language instructions as a prompt (Along 'sub-sequence' axis)
            Note that this (2) is the way how VIMA formulates visual imitation learning.
    """
    PARAM_COMPONENTS = ["policy"]

    def __str__(self):
        return "VIMA"

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        super().__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.embed_dim = cfg["embed_dim"]
        self.prompt_dim = cfg["prompt_dim"]
        self.xf_num_layers = cfg["xf_num_layers"]
        self.sattn_num_heads = cfg["sattn_num_heads"]
        self.xattn_num_heads = cfg["xattn_num_heads"]

        self.policy = None

        if init_build_model:
            self.build_model()

    def build_model(self):
        b = 3
        l = 7
        init_obs = np.zeros((b, l, self.observation_dim))
        init_act = np.zeros((b, l, self.action_dim))
        init_rtg = np.zeros((b, l, 1))
        init_prompt = np.zeros((b, 4, self.prompt_dim))
        init_prompt_maskings = np.ones((b, 4))
        init_seq = np.zeros((b, self.sequential_requirements_dim))
        init_nf = np.zeros((b, self.nonfunc_dim))
        init_prm = np.zeros((b, 4, self.total_param_dim))
        init_timesteps = np.zeros((b, l), dtype="i4")
        maskings = np.ones((b, l))

        self.rng, rngs = get_basic_rngs(self.rng)
        tx = optax.adam(self.cfg["lr"])
        self.policy = Model.create(
            model_def=_VIMA(
                embed_dim=self.embed_dim,
                prompt_dim=self.prompt_dim,
                xf_num_layers=self.xf_num_layers,
                sattn_num_heads=self.sattn_num_heads,
                xattn_num_heads=self.xattn_num_heads,
                rng=self.rng,
            ),
            inputs=[
                rngs,
                init_obs,
                init_act,
                init_rtg,
                init_prompt,
                init_prompt_maskings,
                init_seq,
                init_nf,
                init_prm,
                init_timesteps,
                maskings,
                False
            ],
            tx=tx
        )

    def get_param_for_skills(self, replay_data: ComDeBufferSample):
        skill_param_dict = get_episodic_level_skills(replay_data, param_repeats=self.param_repeats)
        return skill_param_dict["param_for_source_skills"]

    def get_prompts(self, replay_data: ComDeBufferSample):
        source_skills = replay_data.source_skills
        n_source_skills = replay_data.n_source_skills.reshape(-1, 1)

        prompts = []
        for source_skills_idx in replay_data.source_skills_idxs:
            tmp_prompts = np.array([random.choice(self.firstimage_mapping[str(sk)]) for sk in source_skills_idx])
            prompts.append(tmp_prompts)
        prompts = np.array(prompts)

        batch_size = source_skills.shape[0]
        prompts_maskings = np.arange(source_skills.shape[1]).reshape(1, -1)  # [1, M]
        prompts_maskings = np.repeat(prompts_maskings, repeats=batch_size, axis=0)  # [b, M]
        prompts_maskings = np.where(prompts_maskings < n_source_skills, 1, 0)

        return prompts, prompts_maskings

    def update(self, replay_data: ComDeBufferSample) -> Dict:

        param_for_source_skills = self.get_param_for_skills(replay_data)
        prompts, prompts_maskings = self.get_prompts(replay_data)

        rtgs = replay_data.rtgs
        rtgs = rtgs.reshape((*rtgs.shape, 1))

        new_policy, info = promptdt_update(
            rng=self.rng,
            policy=self.policy,
            observations=replay_data.observations,
            actions=replay_data.actions,
            rtgs=rtgs,
            prompts=prompts,
            prompts_maskings=prompts_maskings,
            sequential_requirement=replay_data.sequential_requirement,
            non_functionality=replay_data.non_functionality,
            param_for_skills=param_for_source_skills,
            timesteps=replay_data.timesteps,
            maskings=replay_data.maskings
        )
        self.policy = new_policy
        self.rng, _ = jax.random.split(self.rng)
        return info

    def predict(
        self,
        observations: np.ndarray,	# [b, l, d]
        actions: np.ndarray,	# [b, l, d]
        rtgs: np.ndarray,	# [b, l]
        prompts: np.ndarray,	# [b, M, d]
        prompts_maskings: np.ndarray,	# [b, M]
        sequential_requirement: np.ndarray,	# [b, d]
        non_functionality: np.ndarray,	# [b, d]
        param_for_skills: np.ndarray,	# [b, M, total_prm_dim]
        timesteps: np.ndarray,	# [b, l]
        maskings: np.ndarray,	# [b, l]
        to_np: bool = True,
    ) -> np.ndarray:
        rtgs = rtgs[..., np.newaxis]
        self.rng, actions = forward(
            rng=self.rng,
            model=self.model,
            observations=observations,
            actions=actions,
            rtgs=rtgs,
            prompts=prompts,
            prompts_maskings=prompts_maskings,
            sequential_requirement=sequential_requirement,
            non_functionality=non_functionality,
            param_for_skills=param_for_skills,
            timesteps=timesteps,
            maskings=maskings
        )
        actions = actions[:, -1, ...]
        if to_np:
            return np.array(actions)
        else:
            return actions

    def evaluate(self, *args, **kwargs) -> Dict:
        return dict()

    @property
    def model(self) -> Model:
        return self.policy

    def _excluded_save_params(self) -> List:
        return VIMA.PARAM_COMPONENTS

    def _get_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        for component_str in VIMA.PARAM_COMPONENTS:
            component = getattr(self, component_str)
            params_dict[component_str] = component.params
        return params_dict

    def _get_load_params(self) -> List[str]:
        return VIMA.PARAM_COMPONENTS
