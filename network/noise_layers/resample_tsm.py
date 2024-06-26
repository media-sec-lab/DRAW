from torch import nn
from torchaudio import transforms
import numpy as np

class RSTSM(nn.Module):
    def __init__(self, scale_test=None, scale=None):
        super(RSTSM, self).__init__()
        self.original_sample_rate = 16000
        self.scale = scale
        self.scale_test = scale_test
    def rstsm(self, audio):
        if self.scale_test is not None:
            scale = self.scale_test
            print(f"test tsm rate{scale}")
        else:
            scale = np.random.choice(self.scale, 1)[0]
            # print(f"training tsm rate{scale}")

        new_sample_rate = self.original_sample_rate * scale
        resample = transforms.Resample(orig_freq=self.original_sample_rate, new_freq=new_sample_rate).to(audio.device)
        out = resample(audio)
        return out

    def forward(self, rec_audio):
        return self.rstsm(rec_audio.clone())