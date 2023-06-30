import pickle
import random
from typing import Dict, List

import flax.linen as nn
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from transformers.models.t5.tokenization_t5 import T5Tokenizer as Tokenizer

from comde.baselines.algos.updates.vima import vima_update as policy_update
from comde.baselines.architectures.vima import VIMA as _VIMA
from comde.baselines.utils.get_episode_skills import get_episodic_level_skills
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.type_aliases import ComDeBufferSample
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class TestMLP(nn.Module):
    embed_dim: int
    prompt_dim: int
    xf_num_layers: int
    sattn_num_heads: int
    xattn_num_heads: int

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,  # d_o
        observations_mask: jnp.ndarray,
        actions: jnp.ndarray,   # d_a
        prompt: jnp.ndarray,
        prompt_assets: jnp.ndarray,
        prompt_mask: jnp.ndarray,
        prompt_assets_mask: jnp.ndarray,
        deterministic: bool,
    ):
        batch_size, q_len, action_dim = actions.shape

        obs_tokens = nn.Dense(features=self.embed_dim)(observations)
        act_tokens = nn.Dense(features=self.embed_dim)(actions)

        indices = jnp.arange(q_len * 2).reshape(q_len, 2).T.reshape(-1)
        query = jnp.concatenate((obs_tokens, act_tokens), axis=-2)[:, indices, :]

        # prompt_tokens = nn.Dense(features=self.embed_dim)(prompt[..., jnp.newaxis])
        # prompt_assets_tokens = nn.Dense(features=self.embed_dim)(prompt_assets)
        # key = jnp.concatenate([prompt_tokens, prompt_assets_tokens], axis=-2)

        # result = query @ key.transpose((0, 2, 1))
        # result /= jnp.sqrt(self.embed_dim)
        # result @= key
        # result = nn.Dense(features=action_dim)(result)
        # return result[:, ::2, :]
        return nn.Dense(features=action_dim)(query)[:, ::2, :]


class VIMA(BaseLowPolicy):
    """
    VLPromptDT: Video-Language prompt DT
    Input of PromptDT
        1. State-action-reward history (Like DT)
        2. First image of skill & Language instructions as a prompt (Along 'sub-sequence' axis)
            Note that this (2) is the way how VIMA formulates visual imitation learning.
    """
    PARAM_COMPONENTS = ["policy"]
    _PREFIX = "Follow this video:"

    def __str__(self):
        return "VIMA"

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool = True):
        super().__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.embed_dim = cfg["embed_dim"]
        self.prompt_dim = cfg["prompt_dim"]
        self.xf_num_layers = cfg["xf_num_layers"]
        self.sattn_num_heads = cfg["sattn_num_heads"]
        self.xattn_num_heads = cfg["xattn_num_heads"]

        firstimage_path = cfg["firstimage_path"]
        with open(firstimage_path, "rb") as f:
            firstimage_mapping = pickle.load(f)	# type: Dict[Union[str, int], List]

        if -1 in firstimage_mapping.keys() or "-1" in firstimage_mapping.keys():
            raise LookupError("-1 is for the padded mapping. Please modify the code here.")

        # We don't want integer key.
        self.firstimage_mapping = {str(k): v for k, v in firstimage_mapping.items()}
        self.firstimage_mapping["-1"] = [np.zeros((self.prompt_dim,))]

        self.prefix = None
        self.policy = None

        if init_build_model:
            self.build_model()

    def build_model(self):
        b = 3
        l = 7
        init_obs = jnp.zeros((b, l, self.observation_dim))
        init_act = jnp.zeros((b, l, self.action_dim))
        init_prompt = jnp.zeros((b, 4))
        init_prompt_assets = jnp.zeros((b, 4, self.prompt_dim))
        init_prompt_maskings = jnp.ones((b, 4))
        init_prompt_assets_maskings = jnp.ones((b, 4))
        maskings = jnp.ones((b, l))

        tx = optax.adam(0.1)
        self.rng, rngs = get_basic_rngs(self.rng)
        self.rng, dist_key = jax.random.split(self.rng, 2)
        rngs = {**rngs, "dist": dist_key}
        self.policy = Model.create(
            model_def=_VIMA(
                embed_dim=self.embed_dim,
                prompt_dim=self.prompt_dim,
                xf_num_layers=self.xf_num_layers,
                sattn_num_heads=self.sattn_num_heads,
                xattn_num_heads=self.xattn_num_heads,
            ),
            inputs=[
                rngs,
                init_obs,
                maskings,
                init_act,
                init_prompt,
                init_prompt_assets,
                init_prompt_maskings,
                init_prompt_assets_maskings,
                False,
            ],
            tx=tx
        )

        tokenizer = Tokenizer.from_pretrained("t5-base")
        self.prefix = tokenizer(self._PREFIX, return_tensors="np").input_ids[:, :-1]

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

    def get_prefix_from_prompts(self, prompts: np.ndarray):
        batch_size = prompts.shape[0]
        prefix = np.tile(self.prefix, (batch_size, 1))
        prefix_maskings = np.ones_like(prefix)
        return prefix, prefix_maskings

    def update(self, replay_data: ComDeBufferSample) -> Dict:
        prompts, prompts_maskings = self.get_prompts(replay_data)
        prefix, prefix_maskings = self.get_prefix_from_prompts(prompts)

        rtgs = replay_data.rtgs
        rtgs = rtgs.reshape((*rtgs.shape, 1))

        new_policy, info = policy_update(
            policy=self.policy,
            rng=self.rng,
            observations=replay_data.observations,
            maskings=replay_data.maskings,
            actions=replay_data.actions,
            prompts=prefix,
            prompt_assets=prompts,
            prompts_maskings=prefix_maskings,
            prompt_assets_maskings=prompts_maskings,
        )
        self.rng, _ = jax.random.split(self.rng)
        self.policy = new_policy
        return info

    def predict(
        self,
        observations: np.ndarray,	# [b, l, d]
        actions: np.ndarray,	# [b, l, d]
        rtgs: np.ndarray,	# [b, l]
        prompts: np.ndarray,	# [b, M, d]
        prompts_maskings: np.ndarray,	# [b, M]
        maskings: np.ndarray,	# [b, l]
        to_np: bool = True,
    ) -> np.ndarray:
        rtgs = rtgs[..., np.newaxis]
        self.rng, _ = jax.random.split(self.rng)
        _, num_actions, *_ = actions.shape
        prefix, prefix_maskings = self.get_prefix_from_prompts(prompts)
        actions = self.model.apply_fn(
            observations=observations,
            observations_mask=maskings,
            actions=actions,
            prompts=prefix,
            prompt_assets=prompts,
            prompts_maskings=prefix_maskings,
            prompt_assets_maskings=prompts_maskings,
            deterministic=True,
        )
        actions = actions[:, num_actions - 1, ...]
        if to_np:
            return np.array(actions)
        else:
            return actions

    def evaluate(self, *args, **kwargs) -> Dict:
        return dict()

    @property
    def model(self):
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
