import torch
import numpy as np

def random_translate(imgs, pad=8):
	n, c, h, w = imgs.size()
	imgs = torch.nn.functional.pad(imgs, (pad, pad, pad, pad))
	w1 = torch.randint(0, 2*pad + 1, (n,))
	h1 = torch.randint(0, 2*pad + 1, (n,))
	cropped = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
	for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
		cropped[i][:] = img[:, h11:h11 + h, w11:w11 + w]
	return cropped

class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape)).clip(min_action, max_action)
        return action
