from typing import Any

import numpy as np
import torch as th
from torchvision.models import resnet50, ResNet50_Weights

_weights = None
_resnet50_model = None  # type: Any
_resnet50_preprocess = None  # type: Any


def resnet50_forward(img: np.ndarray) -> np.ndarray:

	if _weights is None:
		_init_pretrained_model()

	# img: [batch, channel, h, w] or [batch, h, w, channel]
	assert img.ndim == 4
	channel_axis = np.argmin(img.shape)
	if channel_axis != 1:  # If channel-last
		img = np.moveaxis(img, source=-1, destination=1)

	img = th.tensor(img.copy(), device="cuda:0")
	img = _resnet50_preprocess(img)
	emb = _resnet50_model(img).cpu().detach().numpy()
	return emb

def _init_pretrained_model():
	global _weights
	global _resnet50_model
	global _resnet50_preprocess

	_weights = ResNet50_Weights.DEFAULT
	_resnet50_model = resnet50(weights=_weights).cuda()
	_resnet50_model.eval()
	_resnet50_preprocess = _weights.transforms()
