import torch
import torch.nn as nn


class TS(nn.Module):
    def __init__(self, length=50):
        super(TS, self).__init__()
        self.shift_length = length

    def time_shift(self, audio, m):
        noised_audio = torch.zeros(audio.shape).to(audio.device)
        for i in range(audio.shape[0]):
            out = torch.cat(
                (audio[i, -m:], audio[i, :-m]), dim=0
            )
            noised_audio[i] = out
        return noised_audio

    def forward(self, rec_audio):
        noised_audio = self.time_shift(rec_audio, self.shift_length)
        return noised_audio
