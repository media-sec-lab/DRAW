import torch
import torch.nn as nn
import numpy as np


class Crop_new(nn.Module):
    def __init__(self, ratio=0.1, loc=[-1,0,1]):
        super(Crop_new, self).__init__()
        self.ratio = ratio
        self.loc = loc

    def crop(self, audios):
        audio_list = audio_split(audios)
        # print("training original crop 0.3")
        crop_audio_list = []
        pad_audio_list = []

        for audio in audio_list:
            # audio [b, frame_length]  一帧
            crop_num = int(audio.shape[-1] * self.ratio)
            crop_loc = np.random.choice(self.loc,1)[0]
            if crop_loc == -1:
                # print("left")
                for i in range(audio.shape[0]):
                    cropped_audio = audio[i, crop_num:]
                    j = i+1 if i<audio.shape[0]-1 else i-1
                    noise = audio[j, :crop_num]
                    padded_audio = torch.cat((noise, cropped_audio)).unsqueeze(0)
                    cropped_audio = cropped_audio.unsqueeze(0)
                    # print(cropped_audio.shape)
                    if i == 0:
                        crop_audio = cropped_audio
                        pad_audio = padded_audio
                    else:
                        crop_audio = torch.cat((crop_audio, cropped_audio), 0)
                        pad_audio = torch.cat((pad_audio, padded_audio), 0)
                crop_audio_list.append(crop_audio.to(audio.device))
                pad_audio_list.append(pad_audio.to(audio.device))

            elif crop_loc == 0:
                # print("middle")
                index = [x for x in range(crop_num, audio.shape[-1] - 2*crop_num - 1)]
                for i in range(audio.shape[0]):
                    mid_crop_begin = np.random.choice(index, 1, replace=False)[0]
                    # print(mid_crop_begin)
                    j = i+1 if i<audio.shape[0]-1 else i-1
                    noise = audio[j, :crop_num]

                    audio_fore = audio[i,: mid_crop_begin]
                    # print(audio_fore.shape)
                    audio_last = audio[i,mid_crop_begin+crop_num: ]
                    # print(audio_last.shape)
                    cropped_audio = torch.cat((audio_fore, audio_last)).unsqueeze(0)
                    padded_audio = torch.cat((audio_fore, noise, audio_last)).unsqueeze(0)
                    if i == 0:
                        crop_audio = cropped_audio
                        pad_audio = padded_audio
                    else:
                        crop_audio = torch.cat((crop_audio, cropped_audio), 0)
                        pad_audio = torch.cat((pad_audio, padded_audio), 0)
                crop_audio_list.append(crop_audio.to(audio.device))
                pad_audio_list.append(pad_audio.to(audio.device))

            else:
                # print("right")
                for i in range(audio.shape[0]):
                    cropped_audio = audio[i, 0: audio.shape[-1] - crop_num]
                    j = i+1 if i<audio.shape[0]-1 else i-1
                    noise = audio[j, :crop_num]
                    padded_audio = torch.cat((cropped_audio, noise)).unsqueeze(0)
                    cropped_audio = cropped_audio.unsqueeze(0)

                    # print(cropped_audio.shape)
                    if i == 0:
                        crop_audio = cropped_audio
                        pad_audio = padded_audio
                    else:
                        crop_audio = torch.cat((crop_audio, cropped_audio), 0)
                        pad_audio = torch.cat((pad_audio, padded_audio), 0)
                crop_audio_list.append(crop_audio.to(audio.device))
                pad_audio_list.append(pad_audio.to(audio.device))

        # print(crop_audio_list[0].shape, pad_audio_list[0].shape)
        return [crop_audio_list, pad_audio_list]
        # return torch.cat(noised_audio_list,dim=-1)

    def forward(self, audio):
        out = self.crop(audio)
        return out

class Crop_new_fix(nn.Module):
    def __init__(self, ratio=0.1, loc=[-1,0,1]):
        super(Crop_new_fix, self).__init__()
        self.ratio = ratio
        self.loc = loc

    def crop(self, audios):
        audio_list = audio_split(audios)
        # print("training original crop 0.3")
        crop_audio_list = []
        pad_audio_list = []

        for audio in audio_list:
            # audio [b, frame_length]  一帧

            crop_num = int(audio.shape[-1] * 0.1)
            index = [x for x in range(crop_num, audio.shape[-1] - 2 * crop_num - 1)]
            for i in range(audio.shape[0]):
                mid_crop_begin = np.random.choice(index, 1, replace=False)[0]
                # print(mid_crop_begin)
                j = i+1 if i<audio.shape[0]-1 else i-1
                noise = audio[j, :crop_num]

                # audio_row = audio[i].clone()
                audio_fore = audio[i][crop_num: mid_crop_begin]
                # print(audio_fore.shape)
                audio_last = audio[i][mid_crop_begin + crop_num: audio.shape[-1] - crop_num]
                # print(audio_last.shape)
                cropped_audio = torch.cat((audio_fore, audio_last)).unsqueeze(0)
                padded_audio = torch.cat((noise, audio_fore, noise, audio_last, noise)).unsqueeze(0)
                if i == 0:
                    crop_audio = cropped_audio
                    pad_audio = padded_audio
                else:
                    crop_audio = torch.cat((crop_audio, cropped_audio), 0)
                    pad_audio = torch.cat((pad_audio, padded_audio), 0)
            crop_audio_list.append(crop_audio.to(audio.device))
            pad_audio_list.append(pad_audio.to(audio.device))

        # print(crop_audio_list[0].shape, pad_audio_list[0].shape)
        return [crop_audio_list, pad_audio_list, "crop"]
        # return torch.cat(noised_audio_list,dim=-1)

    def forward(self, audio):
        out = self.crop(audio)
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
