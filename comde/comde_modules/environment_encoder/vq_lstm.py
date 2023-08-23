import random
from copy import deepcopy
from typing import Dict, Union, List, Tuple

import numpy as np
import optax
from flax import linen as nn

from comde.comde_modules.common.utils import create_mlp
from comde.comde_modules.environment_encoder.algos.forwards import (
	dynamics_encoder_forward as encoder_forward,
	dynamics_decoder_forward as decoder_forward
)
from comde.comde_modules.environment_encoder.algos.vqlstm_update import (
	vqlstm_contrastive_update as contrastive_update,
	vq_lstm_update
)
from comde.comde_modules.environment_encoder.architectures.vq_lstm import PrimVecQuantizedLSTM
from comde.comde_modules.environment_encoder.base import BaseEnvEncoder
from comde.comde_modules.low_policies.base import BaseLowPolicy
from comde.rl.buffers.buffers.comde_buffer import ComdeBuffer, ComDeBufferSample
from comde.utils.interfaces import IJaxSavable, ITrainable
from comde.utils.jax_utils.general import get_basic_rngs
from comde.utils.jax_utils.model import Model
from comde.utils.jax_utils.type_aliases import Params


class VecQuantizedLSTM(BaseEnvEncoder, IJaxSavable, ITrainable):
	PARAM_COMPONENTS = ["encoder"]

	def __init__(
		self,
		seed: int,
		cfg: Dict,
		init_build_model: bool = True
	):
		super(VecQuantizedLSTM, self).__init__(
			seed=seed,
			cfg=cfg,
			init_build_model=init_build_model
		)

		self.encoder = None  # type: Union[nn.Module, Model]
		self.decoder = None  # type: Union[None, nn.Module, Model]
		self.n_codebook = self.cfg["encoder_cfg"]["n_codebook"]
		self.batch_size = cfg["batch_size"]
		self.wind_axis = cfg["wind_axis"]

		from comde.trainer.comde_trainer import ComdeTrainer
		self._get_skill = ComdeTrainer.get_skill_from_idxs

		self.winds = [-0.3, -0.15, 0.0, 0.15, 0.3]
		# self.winds = [-.6, -.3, 0.0, .3, .6]

		if init_build_model:
			self.build_model()

	@property
	def model(self) -> Model:
		raise NotImplementedError()

	def build_model(self):
		encoder = PrimVecQuantizedLSTM(**self.cfg["encoder_cfg"])
		b = 1
		x = np.zeros((b, self.subseq_len, self.observation_dim + self.action_dim))
		self.rng, rngs = get_basic_rngs(self.rng)
		rngs.update({"init_carry": self.rng, "sampling": self.rng + 1})

		encoder_tx = optax.adam(self.cfg["encoder_lr"])
		self.encoder = Model.create(
			model_def=encoder,
			inputs=[rngs, x, 1],
			tx=encoder_tx
		)

		hidden_dim = self.cfg["encoder_cfg"]["lstm_cfg"]["hidden_dim"]
		hiddens = np.zeros((1, hidden_dim))
		states = np.zeros((1, self.observation_dim))
		decoder = create_mlp(**self.cfg["decoder_cfg"])

		decoder_input = np.concatenate((hiddens, states, states), axis=-1)
		self.rng, rngs = get_basic_rngs(self.rng)

		decoder_tx = optax.adam(self.cfg["decoder_lr"])
		self.decoder = Model.create(model_def=decoder, inputs=[rngs, decoder_input], tx=decoder_tx)

	def contrastive_update(self, replay_buffer: ComdeBuffer) -> Dict:
		batch_inds = np.random.choice(len(replay_buffer), size=(len(self.winds),), replace=False)

		# with replay_buffer.history_mode():
		data1 = replay_buffer.sample(batch_inds=batch_inds)
		data2 = replay_buffer.sample(batch_inds=batch_inds)

		first_history_act = data1.actions
		second_history_act = data2.actions

		winds = deepcopy(self.winds)
		random.shuffle(winds)
		for i, wind in enumerate(winds):
			first_history_act[i, ..., self.wind_axis] += wind
			second_history_act[i, ..., self.wind_axis] += wind

		new_encoder, info = contrastive_update(
			rng=self.rng,
			encoder=self.encoder,
			first_history_obs=data1.observations,
			second_history_obs=data2.observations,
			first_history_act=first_history_act,
			second_history_act=second_history_act,
			first_lstm_n_iter=np.sum(data1.maskings, axis=-1),
			second_lstm_n_iter=np.sum(data2.maskings, axis=-1),
			coef_positive_loss=self.cfg["coef_positive_loss"],
			coef_negative_loss=self.cfg["coef_negative_loss"]
		)
		self.encoder = new_encoder
		return info

	def encoder_update(
		self,
		replay_buffer: ComdeBuffer,
		low_policy: BaseLowPolicy,
		last_onehot_skills: Dict
	) -> Tuple[Dict, ComDeBufferSample]:
		with replay_buffer.history_mode():
			replay_data = replay_buffer.sample(batch_size=self.batch_size)

		winds = np.array(random.choices(self.winds, k=self.batch_size))
		nonstationary_actions = replay_data.actions
		nonstationary_actions[..., self.wind_axis] += winds[..., np.newaxis]
		replay_data = replay_data._replace(actions=nonstationary_actions)

		replay_data = self._get_skill(replay_data=replay_data, last_onehot_skills=last_onehot_skills)
		assert replay_data.online_context is None, "Here, the online context should be None."
		skills_dict = low_policy.get_parameterized_skills(replay_data)
		skills = skills_dict["parameterized_skills"]
		lstm_n_iter = np.sum(replay_data.maskings[:, :-1], axis=-1)

		encoder, decoder, info = vq_lstm_update(
			rng=self.rng,
			task_encoder=self.encoder,
			task_decoder=self.decoder,
			low_policy=low_policy.model,
			observations=replay_data.observations,
			actions=replay_data.actions,
			skills=skills,
			lstm_n_iter=lstm_n_iter,
			n_codebook=self.n_codebook,
			coef_skill_dec_aid=self.cfg["coef_skill_dec_aid"],
			coef_commitment=self.cfg["coef_commitment"],
			coef_gamma_moving_avg=self.cfg["coef_gamma_moving_avg"],
			coef_dyna_decoder=self.cfg["coef_dyna_decoder"]
		)
		codebook_idxs = info["__codebook_idxs"]

		n_used_codebook = len(set(np.array(codebook_idxs.astype("i4"))))
		info["dyna_encoder/n_used_cb"] = n_used_codebook

		self.encoder = encoder
		self.decoder = decoder
		return info, replay_data

	def update(
		self,
		replay_buffer: ComdeBuffer,
		low_policy: BaseLowPolicy,
		last_onehot_skills: Dict,
	) -> Dict:
		if self.cfg["use_contrastive"]:
			contrastive_info = self.contrastive_update(replay_buffer)
		else:
			contrastive_info = {}

		encoder_info, replay_data = self.encoder_update(
			replay_buffer=replay_buffer,
			low_policy=low_policy,
			last_onehot_skills=last_onehot_skills
		)
		info = {**contrastive_info, **encoder_info}
		return {"encoder_info": info, "replay_data": replay_data}

	def predict(self, sequence: np.ndarray, n_iter: np.ndarray, deterministic: bool = True) -> np.ndarray:
		rng, info = encoder_forward(
			rng=self.rng,
			encoder=self.encoder,
			sequence=sequence,
			n_iter=n_iter,
			deterministic=deterministic
		)
		self.rng = rng
		return info["quantized"]

	def evaluate(self, replay_buffer: ComDeBufferSample) -> Dict:
		assert replay_buffer.observations.shape[1] == (self.subseq_len + 1), \
			"This requires the history mode. Please sample the replay buffer with history_mode using ComdeBuffer context"

		history_observations = replay_buffer.observations[:, :-1, ...]
		history_actions = replay_buffer.actions[:, :-1, ...]
		history_next_observations = replay_buffer.next_observations[:, :-1, ...]
		encoder_input = np.concatenate((history_observations, history_actions), axis=-1)

		n_iter = np.sum(replay_buffer.maskings[:, :-1], axis=-1)

		quantized = self.predict(sequence=encoder_input, n_iter=n_iter)
		rep_quantized = np.repeat(quantized[:, np.newaxis, ...], repeats=self.subseq_len, axis=1)

		decoded_transitions = decoder_forward(
			rng=self.rng,
			decoder=self.decoder,
			context=rep_quantized,
			history_observations=history_observations,
			history_next_observations=history_next_observations,
			deterministic=True
		)
		decoder_loss = np.mean((decoded_transitions - history_actions) ** 2)

		eval_info = {
			"dyna_encoder/mse_error": decoder_loss,
			"dyna_encoder/mse_error(x100)": decoder_loss * 100,
			"__quantized": quantized
		}
		return eval_info

	def _excluded_save_params(self) -> List:
		return VecQuantizedLSTM.PARAM_COMPONENTS

	def _get_save_params(self) -> Dict[str, Params]:
		params_dict = {}
		for component_str in VecQuantizedLSTM.PARAM_COMPONENTS:
			component = getattr(self, component_str)
			params_dict[component_str] = component.params
		return params_dict

	def _get_load_params(self) -> List[str]:
		return VecQuantizedLSTM.PARAM_COMPONENTS
