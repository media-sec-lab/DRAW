# import librosa
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

is_pytorch_17plus = True

class MFCCLoss(torch.nn.Module):
    def __init__(self,
                 n_fft=2048,
                 win_length=None,
                 hop_length = 512,
                 n_mels=256,
                 n_mfcc = 256,
    ):
        super().__init__()
        self.mfcc_transform = T.MFCC(
                sample_rate=16000,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
        )

    def forward(self, input1, input2):
        # 计算MFCC特征
        mfcc1 = self.mfcc_transform(input1)
        mfcc2 = self.mfcc_transform(input2)
        # print(mfcc1.shape)
        # 计算余弦相似度
        mfcc1 = mfcc1.contiguous().view(mfcc1.size(0), -1)
        mfcc2 = mfcc2.contiguous().view(mfcc2.size(0), -1)
        mfcc1_norm = F.normalize(mfcc1, p=2, dim=1)
        mfcc2_norm = F.normalize(mfcc2, p=2, dim=1)
        cosine_sim = F.cosine_similarity(mfcc1_norm, mfcc2_norm)

        # 返回余弦相似度的负值作为损失
        loss = 1 - cosine_sim.mean()

        return loss


class CustomMSELoss(torch.nn.Module):
    def __init__(self, weight=5, size_average=None, reduce=None, reduction='mean'):
        super(CustomMSELoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, input, target):
        # 计算输入和目标之间的欧氏距离
        # 计算加权 MSE 损失
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        weight_tensor = torch.ones(input.shape).to(input.device)
        for i in range(10):
            weight_tensor.data[:,31+64*(i):64+64*(i)]=self.weight
        # print(weight_tensor[0])
        if weight_tensor is not None:
            diff = (input - target) ** 2
            # print(diff.shape)
            weighted_diff = weight_tensor * diff
            distance = torch.sum(weighted_diff, dim=1)
            loss = distance/input.size(1)
        else:
            distance = torch.norm(input - target, p=2, dim=1)
            loss = (distance ** 2)/input.size(1)
            # print(loss)
        # 根据 reduce 和 reduction 参数决定是否返回标量或张量形式的损失值
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean' and self.size_average:
            return loss.mean()
        elif self.reduction == 'mean':
            return loss.mean(dim=0)
        elif self.reduction == 'sum':
            return loss.sum()

if __name__ == '__main__':
    l = CustomMSELoss()
    l1 = torch.nn.MSELoss()
    x1 = torch.ones([12,640])
    x2 = torch.zeros([12,640])
    x = l(x1,x2)
    x1 = l1(x1,x2)
    print(x, x1)


