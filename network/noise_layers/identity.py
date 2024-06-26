import torch
import torch.nn as nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the audio
	"""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, rec_audio):
		noised_audio = rec_audio
		return noised_audio

