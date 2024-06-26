import torch
import torch.nn as nn
import numpy as np


class Crop(nn.Module):
    def __init__(self, add_noise_choice=None, ratio_max=None,  ratio_test=False, noise=None, replace=None,dis=''):
        super(Crop, self).__init__()
        self.ratio_max = ratio_max
        self.ratio_noise_list = [0.1, 0.1]
        self.ratio_test = ratio_test
        self.noise = noise
        self.add_noise_choice = add_noise_choice
        self.replace = replace
    # crop the audio and add noise
    def crop_noise(self, audio):
        # noised_audio = torch.zeros_like(audio.shape).to(audio.deivce)
        # noised_audio = torch.zeros(1,90)

        loc = np.random.choice(self.add_noise_choice, 1)[0]
        ratio = np.random.uniform(self.ratio_noise_list[0], self.ratio_noise_list[1], 1)[0]

        if loc == 1:
            crop_num = int(audio.shape[-1] * ratio)
            # print(f'add noise crop choice:{ratio} loc:{loc}')
            for i in range(audio.shape[0]):
                audio_fore = audio[i, 0: audio.shape[-1]-crop_num]
                noise = torch.randn(crop_num).to(audio.device)
                cropped_audio = torch.cat((audio_fore, noise)).unsqueeze(0)
                # print(cropped_audio.shape)
                if i == 0:
                    noised_audio = cropped_audio
                else:
                    noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        elif loc == -1:
            crop_num = int(audio.shape[-1] * ratio)
            # print(f'add noise crop choice:{ratio} loc:{loc}')
            for i in range(audio.shape[0]):
                audio_fore = audio[i, crop_num:]
                noise = torch.randn(crop_num).to(audio.device)
                cropped_audio = torch.cat((noise, audio_fore)).unsqueeze(0)
                # print(cropped_audio.shape)
                if i == 0:
                    noised_audio = cropped_audio
                else:
                    noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        # elif ratio == 1.1:
        #     crop_num = int(audio.shape[-1] * 0.1)
        #     print(f'add noise crop choice:{ratio}')
        #
        #     for i in range(audio.shape[0]):
        #         audio_fore = audio[i, 0: audio.shape[-1]-crop_num]
        #         noise = torch.randn(crop_num*2).to(audio.device)
        #         cropped_audio = torch.cat((audio_fore, noise)).unsqueeze(0)
        #         print(cropped_audio.shape)
        #         if i == 0:
        #             noised_audio = cropped_audio
        #         else:
        #             noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        #
        # elif ratio == 0.1:
        #     crop_num = int(audio.shape[-1] * 0.1)
        #     print(f'add noise crop choice:{ratio}')
        #
        #     for i in range(audio.shape[0]):
        #         audio_fore = audio[i, crop_num:]
        #         noise = torch.randn(crop_num).to(audio.device)
        #         cropped_audio = torch.cat((noise, audio_fore)).unsqueeze(0)
        #         print(cropped_audio.shape)
        #         if i == 0:
        #             noised_audio = cropped_audio
        #         else:
        #             noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        #
        # elif ratio == -0.1:
        #     crop_num = int(audio.shape[-1] * 0.1)
        #     print(f'add noise crop choice:{ratio}')
        #
        #     for i in range(audio.shape[0]):
        #         audio_fore = audio[i, :]
        #         noise = torch.randn(crop_num).to(audio.device)
        #         cropped_audio = torch.cat((noise, audio_fore)).unsqueeze(0)
        #         print(cropped_audio.shape)
        #         if i == 0:
        #             noised_audio = cropped_audio
        #         else:
        #             noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        # print(noised_audio.shape)
        return noised_audio.to(audio.device)


    def crop(self, audios):
        # noised_audio = torch.zeros_like(audio.shape).to(audio.deivce)
        if self.ratio_test is not None:
            if self.ratio_test == 0:
                ratio = np.random.uniform(1e-9,self.ratio_max, 1)[0]
                print(f"random ratio :{ratio}")
            else:
                ratio = self.ratio_test
            # audio = input
            if ratio == 0.1:
                crop_num = int(audios.shape[-1] * ratio)
                index = [x for x in range(0, audios.shape[-1] - crop_num - 1)]
                for i in range(audios.shape[0]):
                    mid_crop_begin = np.random.choice(index, 1, replace=False)[0]
                    # print(f"start crop{mid_crop_begin}")
                    audio_row = audios[i].clone()
                    audio_fore = audio_row[0: mid_crop_begin]
                    # print(audio_fore.shape)
                    audio_last = audio_row[mid_crop_begin + crop_num:]
                    # print(audio_last.shape)
                    cropped_audio = torch.cat((audio_fore, audio_last)).unsqueeze(0)
                    if i == 0:
                        noised_audio = cropped_audio
                    else:
                        noised_audio = torch.cat((noised_audio, cropped_audio), 0)
                if self.ratio_test is not None:
                    print(f"testing: start crop num {mid_crop_begin}, ratio: {self.ratio_test}")
                return noised_audio.to(audios.device)
            elif ratio == 0.2:
                crop_num = int(audios.shape[-1] * 0.1)
                index_1 = [x for x in range(0, audios.shape[-1] // 2 - crop_num)]
                index_2 = [x for x in range(audios.shape[-1] // 2, audios.shape[-1] - crop_num)]
                for i in range(audios.shape[0]):
                    crop_1 = np.random.choice(index_1, 1, replace=False)[0]
                    crop_2 = np.random.choice(index_2, 1, replace=False)[0]
                    # print(f"start crop{mid_crop_begin}")
                    audio_row = audios[i].clone()
                    audio_fore = audio_row[0: crop_1]
                    # print(audio_fore.shape)
                    audio_mid = audio_row[crop_1 + crop_num: crop_2]
                    audio_last = audio_row[crop_2 + crop_num:]
                    # print(audio_last.shape)
                    cropped_audio = torch.cat((audio_fore, audio_mid, audio_last)).unsqueeze(0)
                    if i == 0:
                        noised_audio = cropped_audio
                    else:
                        noised_audio = torch.cat((noised_audio, cropped_audio), 0)
                if self.ratio_test is not None:
                    print(f"testing: start crop num {crop_1} {crop_2}, ratio: {self.ratio_test}")
                return noised_audio.to(audios.device)

            elif ratio == 0.3:
                crop_num = int(audios.shape[-1] * 0.1)
                index_1 = [x for x in range(0, audios.shape[-1] // 3 - crop_num)]
                index_2 = [x for x in range(audios.shape[-1] // 3, audios.shape[-1] * 2 // 3 - crop_num)]
                index_3 = [x for x in range(audios.shape[-1] * 2 // 3, audios.shape[-1] - crop_num)]
                for i in range(audios.shape[0]):
                    crop_1 = np.random.choice(index_1, 1, replace=False)[0]
                    crop_2 = np.random.choice(index_2, 1, replace=False)[0]
                    crop_3 = np.random.choice(index_3, 1, replace=False)[0]
                    # print(f"start crop{mid_crop_begin}")
                    audio_row = audios[i].clone()
                    audio_fore = audio_row[0: crop_1]
                    # print(audio_fore.shape)
                    audio_mid_1 = audio_row[crop_1 + crop_num: crop_2]
                    audio_mid_2 = audio_row[crop_2 + crop_num: crop_3]
                    audio_last = audio_row[crop_3 + crop_num:]
                    # print(audio_last.shape)
                    cropped_audio = torch.cat((audio_fore, audio_mid_1, audio_mid_2, audio_last)).unsqueeze(0)
                    if i == 0:
                        noised_audio = cropped_audio
                    else:
                        noised_audio = torch.cat((noised_audio, cropped_audio), 0)
                if self.ratio_test is not None:
                    print(f"testing: start crop num {crop_1} {crop_2} {crop_3}, ratio: {self.ratio_test}")
                return noised_audio.to(audios.device)
        # else:
        #     # print('random crop')
        #     # audio = input
        #     crop_num = int(audio.shape[-1] * 0.1)
        #     index_1 = [x for x in range(0, audio.shape[-1] // 3 - crop_num)]
        #     index_2 = [x for x in range(audio.shape[-1] // 3, audio.shape[-1] * 2 // 3 - crop_num)]
        #     index_3 = [x for x in range(audio.shape[-1] * 2 // 3, audio.shape[-1] - crop_num)]
        #     for i in range(audio.shape[0]):
        #         crop_1 = np.random.choice(index_1, 1, replace=False)[0]
        #         crop_2 = np.random.choice(index_2, 1, replace=False)[0]
        #         crop_3 = np.random.choice(index_3, 1, replace=False)[0]
        #         # print(f"start crop{mid_crop_begin}")
        #         audio_row = audio[i].clone()
        #         audio_fore = audio_row[0: crop_1]
        #         # print(audio_fore.shape)
        #         audio_mid_1 = audio_row[crop_1 + crop_num: crop_2]
        #         audio_mid_2 = audio_row[crop_2 + crop_num: crop_3]
        #         audio_last = audio_row[crop_3 + crop_num:]
        #         # print(audio_last.shape)
        #         cropped_audio = torch.cat((audio_fore, audio_mid_1, audio_mid_2, audio_last)).unsqueeze(0)
        #         if i == 0:
        #             noised_audio = cropped_audio
        #         else:
        #             noised_audio = torch.cat((noised_audio, cropped_audio), 0)
        #     return ("crop",noised_audio.to(audio.device))
            else:
                crop_num = int(audios.shape[-1] * ratio)
                index = [x for x in range(crop_num, audios.shape[-1] - crop_num - 1)]
                for i in range(audios.shape[0]):
                    mid_crop_begin = np.random.choice(index, 1, replace=False)[0]
                    # print(mid_crop_begin)
                    audio_row = audios[i].clone()
                    audio_fore = audio_row[: mid_crop_begin]
                    # print(audio_fore.shape)
                    audio_last = audio_row[mid_crop_begin+crop_num:]
                    # print(audio_last.shape)
                    cropped_audio = torch.cat((audio_fore, audio_last)).unsqueeze(0)
                    if i == 0:
                        noised_audio = cropped_audio
                    else:
                        noised_audio = torch.cat((noised_audio, cropped_audio), 0)
                return noised_audio
            # return torch.cat(noised_audio_list,dim=-1)
    def forward(self, audio):
        if not self.noise:
            # print('really crop')
            out = self.crop(audio)
        else:
            # print('noise crop')
            out = self.crop_noise(audio)
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
# class Crop(nn.Module):
#     def __init__(self, crop_length_ratio=0.1):
#         super(Crop, self).__init__()
#         self.crop_length_ratio = crop_length_ratio
#
#     def crop(self, audio):
#         # noised_audio = torch.zeros_like(audio.shape).to(audio.deivce)
#         # noised_audio = torch.zeros(1,90)
#         crop_num = int(audio.shape[-1] * self.crop_length_ratio)
#         index = [x for x in range(crop_num, audio.shape[-1] - 2*crop_num - 1)]
#         for i in range(audio.shape[0]):
#             mid_crop_begin = np.random.choice(index, 1, replace=False)[0]
#
#             audio_row = audio[i].clone()
#             audio_fore = audio_row[crop_num: mid_crop_begin]
#             # print(audio_fore.shape)
#             audio_last = audio_row[mid_crop_begin+crop_num: audio.shape[-1]-crop_num]
#             # print(audio_last.shape)
#             cropped_audio = torch.cat((audio_fore, audio_last)).unsqueeze(0)
#             if i == 0:
#                 noised_audio = cropped_audio
#             else:
#                 noised_audio = torch.cat((noised_audio, cropped_audio), 0)
#
#         return noised_audio.to(audio.device)
#
#     def forward(self, audio):
#         return self.crop(audio)