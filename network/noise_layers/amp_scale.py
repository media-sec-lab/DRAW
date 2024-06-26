import torch
import torch.nn as nn


class AS(nn.Module):
    def __init__(self, strength=0.85):
        super(AS, self).__init__()
        self.strength = strength

    def forward(self, rec_audio):
        noised_audio = rec_audio.clone() * self.strength
        return noised_audio



