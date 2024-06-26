import math
import torch
import torch.nn as nn
import soundfile as sf

# class GN(nn.Module):
#
#     def __init__(self, strength=30):
#         super(GN, self).__init__()
#         self.strength = strength  # db
#
#     def gaussian_noise(self, audio, snr):
#         noise_audio = torch.zeros(audio.shape)
#         for i in range(audio.shape[0]):
#             audio_row = audio[i, :].clone().unsqueeze(0)
#             noise = torch.randn(audio_row.shape).to(audio.device)
#             Ps = torch.sum(audio_row ** 2) / audio_row.shape[1]
#             Pn1 = torch.sum(noise ** 2) / noise.shape[1]
#             k = math.sqrt(Ps / (10 ** (snr / 10) * Pn1))
#             noise = noise * k
#             out = audio_row + noise
#             noise_audio[i, :] = out
#         return noise_audio
#
#     def forward(self, rec_audio):
#         return self.gaussian_noise(rec_audio, self.strength)

# class GN(nn.Module):
#
#     def __init__(self, snr=30):
#         super(GN, self).__init__()
#         self.snr = snr  # dB
#
#     def gaussian_noise(self, audio, snr):
#         noise_audio = torch.zeros_like(audio)
#         for i in range(audio.shape[0]):
#             audio_row = audio[i, :].clone().unsqueeze(0)
#             noise = torch.randn_like(audio_row).to(audio.device)
#             Ps = torch.norm(audio_row) ** 2 / audio_row.shape[1]
#             Pn1 = torch.norm(noise) ** 2 / noise.shape[1]
#             k = math.sqrt(Ps / (10 ** (snr / 10) * Pn1))
#             noise = noise * k
#             out = audio_row + noise
#             noise_audio[i, :] = out
#         return noise_audio
#
#     def forward(self, rec_audio):
#         snr_linear = 10 ** (self.snr / 10)
#         return self.gaussian_noise(rec_audio, snr_linear)

import torch
import torch.nn as nn
import math
import numpy as np
class GN(nn.Module):

    def __init__(self, snr=30, dis=''):
        super(GN, self).__init__()
        self.snr = snr  # dB

    def gaussian_noise(self, audio, snr):
        noise = torch.randn_like(audio).to(audio.device)
        Ps = torch.mean(audio ** 2)
        Pn1 = torch.mean(noise ** 2)
        k = math.sqrt(Ps / (10 ** (snr / 10) * Pn1))
        noise = noise * k
        out = audio + noise
        return out

    def forward(self, rec_audio):
        # snr_linear = 10 ** (self.snr / 10)
        # snr = torch.randint(20, 30, (1,)).item()
        # print(f"noise snr{snr}")
        return self.gaussian_noise(rec_audio, self.snr)

if __name__=="__main__":
    def compute_snr( input_signal, output_signal):
        Ps = torch.sum(torch.abs(input_signal ** 2))
        Pn = torch.sum(torch.abs((input_signal - output_signal) ** 2))
        return 10 * torch.log10((Ps / Pn))
    def compute_snr_batch( input_signals, output_signals):
        snr = 0.0
        batch_size = len(input_signals)
        for i in range(batch_size):
            snr += compute_snr(input_signals[i], output_signals[i])
        snr /= batch_size
        return snr

    gn = GN(20)
    # audio,sr = sf.read("/data/chenjincheng/code/test/long_btach_sp/results/33bit/vctk_500000_16.wav")
    audio,sr = sf.read("/home/linkaiqing/cjc/dataset/p225_118.wav")
    audio =torch.FloatTensor(audio).unsqueeze(0)
    out = gn(audio)
    snr = compute_snr_batch(audio,out)
    print(snr)


