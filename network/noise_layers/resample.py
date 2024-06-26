from torch import nn
from torchaudio import transforms


class RS(nn.Module):
    def __init__(self, down_sample_ratio=0.8,dis=''):
        super(RS, self).__init__()
        self.original_sample_rate = 16000
        self.down_sample_rate = down_sample_ratio * self.original_sample_rate

    def forward(self, rec_audio):
        down_sample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=self.down_sample_rate).to(rec_audio.device)
        up_sample = transforms.Resample(orig_freq=self.down_sample_rate, new_freq=self.original_sample_rate).to(rec_audio.device)
        down_sample_audio = down_sample(rec_audio)
        up_sample_audio = up_sample(down_sample_audio).to(rec_audio.device)
        return up_sample_audio


# class RS(nn.Module):
#     def __init__(self, down_sample_ratio=[0.8,0.9,1.1,1.2]):
#         super(RS, self).__init__()
#         self.original_sample_rate = 16000
#         self.down_sample_ratio = down_sample_ratio
#
#     def forward(self, rec_audio):
#         ratio = np.random.choice(self.down_sample_ratio, 1)[0]
#         if ratio<1:
#             down_sample_rate = ratio * self.original_sample_rate
#             # print(down_sample_rate)
#             down_sample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=down_sample_rate).to(rec_audio.device)
#             out = down_sample(rec_audio.clone())
#         if ratio>1:
#             up_sample_rate = ratio * self.original_sample_rate
#             # print(up_sample_rate)
#             up_sample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=up_sample_rate).to(rec_audio.device)
#             out = up_sample(rec_audio.clone())
#         return out


# class DS(nn.Module):
#     def __init__(self, down_sample_rate=8000):
#         super(DS, self).__init__()
#         self.original_sample_rate = 16000
#         self.down_sample_rate = down_sample_rate
#         self.down_sample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=self.down_sample_rate, lowpass_filter_width=32)
#
#     def forward(self, rec_audio):
#         self.down_sample = self.down_sample.to(rec_audio.device)
#         return self.down_sample(rec_audio.clone())
#
#
# class US(nn.Module):
#     def __init__(self, up_sample_rate=32000):
#         super(US, self).__init__()
#         self.original_sample_rate = 16000
#         self.up_sample_rate = up_sample_rate
#         self.up_sample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=self.up_sample_rate, lowpass_filter_width=32)
#
#
#     def forward(self, rec_audio):
#         self.up_sample = self.up_sample.to(rec_audio.device)
#         return self.up_sample(rec_audio.clone())