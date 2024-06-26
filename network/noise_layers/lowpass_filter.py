import torch
import torch.nn as nn
from torchaudio import functional
from math import pi, sin, cos, sqrt
from cmath import exp
import torch.nn.functional as F


class LF(nn.Module):
    def __init__(self, cut_off_fre=8000, test=None):
        super(LF, self).__init__()
        self.sample_rate = 16000
        self.cut_off = cut_off_fre
        self.test = test
    def low_pass(self, input):
        # filted_audio = torch.zeros(audio.shape).to(audio.device)
        out = []
        for audio in input:
            # print(audio.shape)
            filted_audio = functional.lowpass_biquad(audio.clone(), self.sample_rate, self.cut_off).to(audio.device)
            out.append(filted_audio)
        out = torch.cat(out,dim=-1)
        # print(out.shape)
        return out
    def forward(self, rec_audio):
        if not self.test:
            audio_list = audio_split(rec_audio)
            out = self.low_pass(audio_list)
        else:
            out = functional.lowpass_biquad(rec_audio, self.sample_rate, self.cut_off).to(rec_audio.device)
        return out

def audio_split(audio, segment_num=10):
    total_length = audio.shape[-1]
    # print(total_length)
    seg_audio = []
    seg_audio_length = total_length // segment_num
    for i in range(segment_num):
        # print(i)
        audio_segment = audio[:,(seg_audio_length*i): seg_audio_length*(i+1)]  # [1,16129]
        seg_audio.append(audio_segment)
    return seg_audio





# def bwsk(k, n):
#     # Returns k-th pole s_k of Butterworth transfer
#     # function in S-domain. Note that omega_c
#     # is not taken into account here
#     arg = pi * (2 * k + n - 1) / (2 * n)
#     return complex(cos(arg), sin(arg))
#
# def bwj(k, n):
#     # Returns (s - s_k) * H(s), where
#     # H(s) - BW transfer function
#     # s_k  - k-th pole of H(s)
#     res = complex(1, 0)
#     for m in range(1, n + 1):
#         if (m == k):
#             continue
#         else:
#             res /= (bwsk(k, n) - bwsk(m, n))
#     return res
#
# def bwh(n=16, fc=8000, fs=16000, length=25):
#     # Returns h(t) - BW transfer function in t-domain.
#     # length is in ms.
#     omegaC = 2*pi*fc
#     dt = 1/fs
#     number_of_samples = int(fs*length/1000)
#     result = []
#     for x in range(number_of_samples):
#         res = complex(0, 0)
#         if x >= 0:
#             for k in range(1, n + 1):
#                 res += (exp(omegaC*x*dt/sqrt(2)*bwsk(k, n)) * bwj(k, n))
#         result.append((res).real)
#     return result[::-1]
#
# class LF(nn.Module):
#     def __init__(self):
#         super(LF, self).__init__()
#         self.weight = torch.FloatTensor(bwh()).unsqueeze(0).unsqueeze(0)
#
#     def forward(self, rec_audio):
#         filter_window = self.weight.to(rec_audio.device)
#         filtered_signals = torch.zeros(rec_audio.shape)
#         for i in range(rec_audio.shape[0]):
#             single_audio = rec_audio[i].clone().unsqueeze(0).unsqueeze(0)
#             filtfilt = F.conv1d(single_audio, filter_window, padding="same")
#             filtfilt = filtfilt.squeeze(0)
#             filtered_signals[i] = filtfilt
#         return filtered_signals




# class LF(nn.Module):
#     def __init__(self):
#         super(LF, self).__init__()
#         self.butter_lowpass = butter_lowpass
#         self.butter_lowpass_filt = butter_lowpass_filter
#
#     def forward(self, rec_audio):
#         filtered_signals = []
#         for i in range(len(rec_audio)):
#             # data = rec_audio[i].clone().detach().cpu().numpy()
#             filtfilt = self.butter_lowpass_filt(rec_audio[i].detach().cpu())
#             filt_tensor = torch.FloatTensor(filtfilt.copy()).unsqueeze(dim=0)
#             filt_tensor.requires_grad = True
#             filtered_signals.append(filt_tensor)
#         filtered_signals = torch.cat(filtered_signals, dim=0)
#         # print(filtered_signals.shape)
#
#         return filtered_signals

