import torch
import torch.nn as nn


class Jitter(nn.Module):
    def __init__(self, hop_length=100, test=None):
        super(Jitter, self).__init__()
        self.hop_lenght = hop_length
        self.test = test
    def jittering(self, audio):
        # noised_audio = torch.zeros_like(audio.shape).to(audio.deivce)
        # noised_audio = torch.zeros(1,90)
        num = audio.shape[-1] // self.hop_lenght
        for i in range(audio.shape[0]):
            # audio_row = audio[i].clone()
            audio_res = audio[i][num * self.hop_lenght:].unsqueeze(0)
            # print(f'audio_res{audio_res.shape}')
            audio_cut = audio[i][:num * self.hop_lenght]
            audio_cut = audio_cut.view(-1, self.hop_lenght)
            # print(f'audio_cut{audio_cut.shape}')
            # print(audio_cut.shape)
            audio_delete = audio_cut[:, :self.hop_lenght-1]
            # print(f'audio_delete{audio_delete.shape}')
            out = audio_delete.contiguous().view(1, -1)
            jittered_audio = torch.cat((out, audio_res), dim=1)
            if i == 0:
                noised_audio = jittered_audio
            else:
                noised_audio = torch.cat((noised_audio, jittered_audio), 0)
        if self.test:
            return noised_audio.to(audio.device)
        else:
            return ('jitter',noised_audio.to(audio.device))
            # return noised_audio.to(audio.device)    # 消融

    def forward(self, audio):
        return self.jittering(audio)

# import torch
# import torch.nn as nn
#
# class Jitter(nn.Module):
#     def __init__(self, hop_length=100, test=None):
#         super(Jitter, self).__init__()
#         self.hop_length = hop_length
#         self.test = test
#
#     def jittering(self, audio):
#         num_frames = audio.shape[-1] // self.hop_length  # 计算帧数
#         jittered_audio = torch.zeros_like(audio)  # 创建与输入音频相同形状的零矩阵
#
#         for i in range(audio.shape[0]):
#             for j in range(num_frames):
#                 start = j * self.hop_length
#                 end = (j + 1) * self.hop_length
#                 if end <= audio.shape[-1]:  # 仅当结束索引不超过音频长度时进行操作
#                     if i == 0:
#                         jittered_audio[i][start:end-1] = audio[i][start+1:end]  # 删除每隔100个样本点的第一个样本点
#                     else:
#                         jittered_audio[i][start:end-1] = audio[i][start+1:end]
#
#         if self.test:
#             return jittered_audio.to(audio.device)
#         else:
#             return ('jitter', jittered_audio.to(audio.device))
#
#     def forward(self, audio):
#         return self.jittering(audio)

