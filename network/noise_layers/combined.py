from . import Identity
import torch.nn as nn
from . import get_random_int
import numpy as np
from utils.load_train_setting import attack_pro


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, rec_audio):
		# print(attack_pro)
		id = np.random.choice(len(self.list), 1, p=attack_pro)[0]   # 46
		# print(f'noise layer:{self.list[id]}')
		return self.list[id](rec_audio)
