import math
import torch
import torch.nn as nn
from audiomentations import Mp3Compression


class MP3(nn.Module):

    def __init__(self, bit_rate=64, sample_rate=16000,test=None,dis=''):
        super(MP3, self).__init__()
        self.sample_rate = sample_rate
        self.num_sample = 161290
        self.compress = Mp3Compression(p=1.0, min_bitrate=bit_rate, max_bitrate=bit_rate)
        self.test = test

    def mp3(self, y):
        f = []
        device = y.device
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.compress(i,sample_rate=self.sample_rate)[:self.num_sample]))
        f = torch.stack(f,dim=0).to(device)
        y = y + (f - y)
        return y

    def mp3_test(self, y):
        f = []
        device = y.device
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.compress(i,sample_rate=self.sample_rate)[:a.shape[-1]]))
        f = torch.stack(f,dim=0).to(device)
        y = y + (f - y)
        return y

    def forward(self, rec_audio):
        if self.test:
            return self.mp3_test(rec_audio)
        else:
            return self.mp3(rec_audio)
