import numpy as np
import torch as th
from torchvision.models import resnet50, ResNet50_Weights


_weights = ResNet50_Weights.DEFAULT
_resnet50_model = resnet50(weights=_weights).cuda()
_resnet50_model.eval()
_renset50_preprocess = _weights.transforms()


def resnet50_forward(img: np.ndarray) -> np.ndarray:
	# img: [batch, channel, h, w] or [batch, h, w, channel]
	assert img.ndim == 4
	channel_axis = np.argmin(img.shape)
	if channel_axis != 1:	# If channel-last
		img = np.moveaxis(img, source=-1, destination=1)

	img = th.tensor(img.copy(), device="cuda:0")
	img = _renset50_preprocess(img)
	emb = _resnet50_model(img).cpu().detach().numpy()
	return emb