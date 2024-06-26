from torchaudio import transforms
import torch
import torch.nn as nn
import pytsmod as tsm
import numpy as np


# class TSM(nn.Module):
#
#     def __init__(self, scale_test=None, scale=[0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]):
#         super(TSM, self).__init__()
#         self.scale_factor = scale
#         self.scale_test = scale_test
#
#     def time_scale_modification(self, audio):
#         if self.scale_test is not None:
#             rate = self.scale_test
#         else:
#             rate = np.random.choice(self.scale_factor, 1)[0]
#         device = audio.device
#         audio_clone = audio.clone().detach().cpu()
#         noised_audio = tsm.phase_vocoder(audio_clone, rate)
#         return torch.FloatTensor(noised_audio).to(device)
#
#     def forward(self, rec_audio):
#         if self.scale_test is not None:
#             out = self.time_scale_modification(rec_audio).unsqueeze(0)
#         else:
#             out = self.time_scale_modification(rec_audio)
#
#         return out


class TSM(nn.Module):

    def __init__(self, scale_test=None,scale=None, random=None,dis=''):
        super(TSM, self).__init__()
        self.random = random
        self.scale_factor = scale
        self.scale_test = scale_test

    def time_scale_modification(self, audio):
        if self.scale_test is not None:
            rate = self.scale_test
            print(f"testing tsm rate{rate}")
        elif self.random:
            rate = np.random.uniform(0.8,1.2,1)[0]
            print(f"testing tsm rate{rate}")
        else:
            rate = np.random.choice(self.scale_factor, 1)[0]
            # print(f'tsm rate {rate}')
        device = audio.device
        audio_clone = audio.clone().detach().cpu()
        noised_audio = tsm.phase_vocoder(audio_clone, rate)
        return rate, torch.FloatTensor(noised_audio).to(device)

    def forward(self, rec_audio):
        if self.scale_test is not None:
            rate,out = self.time_scale_modification(rec_audio)
            return out.unsqueeze(0)
        elif self.random:
            rate,out = self.time_scale_modification(rec_audio)
            return out.unsqueeze(0)
        else:
            rate,out = self.time_scale_modification(rec_audio)

            return (rate,out)
            # return out    #消融



# class TSM(nn.Module):
#
#     def __init__(self, rate=0.8):
#         super(TSM, self).__init__()
#         self.rate = rate
#
#     def time_scale_modification(self, audio):
#         device = audio.device
#         audio_clone = audio.clone().detach().cpu()
#         noised_audio = tsm.phase_vocoder(audio_clone, self.rate)
#         return torch.FloatTensor(noised_audio).unsqueeze(0).to(device)  # test
#         # return torch.FloatTensor(noised_audio).to(device)
#
#     def forward(self, rec_audio):
#         return self.time_scale_modification(rec_audio)


