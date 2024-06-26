from torchaudio import transforms
import math
import torch
import torch.nn as nn
from torchaudio import functional as F
from torchaudio import transforms as T
import numpy as np
class PSM(nn.Module):

    def __init__(self, rate=[0.9,1.1],test=None):
        super(PSM, self).__init__()
        self.sample_rate = 16000
        self.rate = rate
        self.test =test
    # def pitch_scale_modification(self, audio):
    #     # trasform = transforms.PitchShift(self.sample_rate, self.step).to(audio.device)
    #     return out

        # noised_audio = F.pitch_shift(audio.clone(), self.sample_rate, self.step).to(audio.device)
        # return noised_audio

    def forward(self, rec_audio):
        if self.test is None:
            rate = np.random.choice(self.rate, 1)[0]
        else:
            rate =self.test
            print(f"testing ps:{rate}")
        return F.pitch_shift(rec_audio, self.sample_rate, 12 * (rate - 1))





