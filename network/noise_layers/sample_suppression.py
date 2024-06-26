import torch
import torch.nn as nn
import numpy as np


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

class SS(nn.Module):
    def __init__(self, attack_rate = 0.03):
        super(SS, self).__init__()
        # self.attacked_sample_index = np.random.choice(self.index_list, self.num_attacked_sample, replace=False)
        self.attack_rate = attack_rate

    def sample_suppression(self, audio):
        noise_audio = torch.zeros(audio.shape).to(audio.device)
        num_sample = audio.shape[-1]
        index_list = [int(d) for d in range(num_sample)]
        num_attacked_sample = int(self.attack_rate * num_sample)  # number of sample to be attacked

        for i in range(audio.shape[0]):
            audio_row = audio[i, :].clone()
            attacked_sample_index = np.random.choice(index_list, num_attacked_sample, replace=False)

            audio_row.scatter_(0, torch.LongTensor(attacked_sample_index).to(audio.device), 0)

            noise_audio[i, :] = audio_row
        return noise_audio

    def forward(self, rec_audio):
        return self.sample_suppression(rec_audio)


class Pad(nn.Module):
    def __init__(self,):
        super(Pad, self).__init__()

    def frame_padding(self,audios):
        audios_frame = audio_split(audios, 10)
        pad_audio_list = []
        for audio_frame in audios_frame:
            pad_length = np.random.randint(0, 4) * 127
            for i in range(audio_frame.shape[0]):
                j = i+1 if i<audio_frame.shape[0]-1 else i-1
                noise = audio_frame[j, :pad_length]
                padded_audio = torch.cat((noise, audio_frame[i], noise)).unsqueeze(0)
                if i == 0:
                    pad_audio = padded_audio
                else:
                    pad_audio = torch.cat((pad_audio, padded_audio), 0)
            pad_audio_list.append(pad_audio.to(audios.device))
        return [audios_frame, pad_audio_list, "pad"]
    def forward(self, audios):
        return self.frame_padding(audios)
