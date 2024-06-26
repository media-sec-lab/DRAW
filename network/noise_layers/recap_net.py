import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

class Generator(nn.Module):
    """G"""

    def __init__(self):
        super().__init__()
        # encoder gets a noisy signal as input [B x 1 x 16384]
        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)  # [B x 16 x 8192]
        self.enc1_nl = nn.PReLU()
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # [B x 64 x 512]
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # [B x 128 x 256]
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # [B x 128 x 128]
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # [B x 256 x 64]
        self.enc8_nl = nn.PReLU()
        self.enc9 = nn.Conv1d(256, 256, 32, 2, 15)  # [B x 256 x 32]
        self.enc9_nl = nn.PReLU()
        self.enc10 = nn.Conv1d(256, 512, 32, 2, 15)  # [B x 512 x 16]
        self.enc10_nl = nn.PReLU()
        self.enc11 = nn.Conv1d(512, 1024, 32, 2, 15)  # [B x 1024 x 8]
        self.enc11_nl = nn.PReLU()

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homologous encoder output,
        # so the feature map sizes are doubled
        self.dec10 = nn.ConvTranspose1d(in_channels=2048, out_channels=512, kernel_size=32, stride=2, padding=15)
        self.dec10_nl = nn.PReLU()  # out : [B x 512 x 16] -> (concat) [B x 1024 x 16]
        self.dec9 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)  # [B x 256 x 32]
        self.dec9_nl = nn.PReLU()
        self.dec8 = nn.ConvTranspose1d(512, 256, 32, 2, 15)  # [B x 256 x 64]
        self.dec8_nl = nn.PReLU()
        self.dec7 = nn.ConvTranspose1d(512, 128, 32, 2, 15)  # [B x 128 x 128]
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # [B x 128 x 256]
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # [B x 64 x 512]
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # [B x 16 x 8192]
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # [B x 1 x 16384]
        self.dec_tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input batch (signal)
            z: latent vector
        """
        # encoding step
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))
        e9 = self.enc9(self.enc8_nl(e8))
        e10 = self.enc10(self.enc9_nl(e9))
        e11 = self.enc11(self.enc10_nl(e10))
        # c = compressed feature, the 'thought vector'
        c = self.enc11_nl(e11)

        # concatenate the thought vector with latent variable
        encoded = torch.cat((c, z), dim=1)

        # decoding step
        d10 = self.dec10(encoded)
        # dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d10_c = self.dec10_nl(torch.cat((d10, e10), dim=1))
        d9 = self.dec9(d10_c)
        d9_c = self.dec9_nl(torch.cat((d9, e9), dim=1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, e8), dim=1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out


class RCT(nn.Module):
    def __init__(self,noise_snr=None):
        super(RCT, self).__init__()
        # self.weight_path_1 = '/data/chenjincheng/code/test/long_btach_sp/results/recap_net/recap_net_weight/G_43_0.5160.pkl'
        self.weight_path_1 = '/raid/chenjincheng/code/long_btach_sp/recap_net_weight/G_86_2.1886.pkl'
        # self.weight_path_1 = '/raid/chenjincheng/code/long_btach_sp/recap_net_weight/G_43_0.5160.pkl'
        # self.weight_path_2 = '/raid/chenjincheng/code/long_btach_sp/recap_net_weight/G_81_1.2047.pkl'
        self.recap_noise_net = Generator().cuda()
        # self.recap_reverb_net = Generator().cuda()

        self.recap_noise_net.load_state_dict(torch.load(self.weight_path_1, map_location='cuda'))
        self.recap_noise_net.eval()
        # self.recap_reverb_net.load_state_dict(torch.load(self.weight_path_2, map_location='cuda'))
        # self.recap_reverb_net.eval()

        for param in self.recap_noise_net.parameters():
            param.requires_grad = False
        # for param in self.recap_reverb_net.parameters():
        #     param.requires_grad = False

        self.noise_path = '/raid/chenjincheng/datasets/noise'
        self.noise_snr = noise_snr
    def emphasis(self, signal, emph_coeff=0.95, pre=True):
        if pre:
            result = torch.cat((signal[0].unsqueeze(0), signal[1:] - emph_coeff * signal[:-1]), dim=-1)
        else:
            result = torch.cat((signal[0].unsqueeze(0), signal[1:] + emph_coeff * signal[:-1]), dim=-1)

        return result

    def recapture(self, audios):
        recap_audio = []
        for audio in audios:
            win_len = 16384
            # 不足的部分 重复填充
            N_slice = len(audio) // win_len
            temp_noisy = audio
            if not len(audio) % win_len == 0:
                short = win_len - len(audio) % win_len
                to_pad = audio[:short]
                temp_noisy = torch.cat((audio, to_pad), dim=-1)
                N_slice = N_slice + 1

            slices = temp_noisy.reshape(N_slice, win_len)

            enh_slice = torch.zeros(slices.shape)

            # 逐帧进行处理
            for n in range(N_slice):
                m_slice = slices[n]   #[16129]
                # 进行预加重
                m_slice = self.emphasis(m_slice)
                # 增加 2个维度
                m_slice = m_slice.unsqueeze(0).unsqueeze(0)

                # 生成 z
                z = nn.init.normal_(torch.Tensor(1, 1024, 8)).cuda()

                # 进行增强
                generated_slice = self.recap_noise_net(m_slice, z)
                # generated_slice = self.recap_reverb_net(generated_slice, z)
                # 反预加重
                generated_slice = self.emphasis(generated_slice[0, 0, :], pre=False)
                enh_slice[n] = generated_slice

            # 信号展开
            enh_speech = enh_slice.reshape(N_slice * win_len)
            recap_audio.append(enh_speech[:len(audio)])

        return torch.stack(recap_audio)

    def compute_snr(self, input_signal, output_signal):
        Ps = torch.sum(torch.abs(input_signal ** 2))
        Pn = torch.sum(torch.abs((input_signal - output_signal) ** 2))
        return 10 * torch.log10((Ps / Pn))


    def add_noise(self, org_audio, noi_audio, snr):
        Ps = torch.sum(org_audio ** 2) / org_audio.shape[0]  # 计算音频信号的功率
        Pn1 = torch.sum(noi_audio ** 2) / noi_audio.shape[0]  # 计算噪声的功率
        # print(f"ps:{Ps}, pn:{Pn1}")
        scalar = torch.sqrt(Ps / (10 ** (snr / 10)) / (Pn1+torch.finfo(torch.float32).eps))

        # 将噪音音频与原始音频叠加
        mixed_audio = org_audio + noi_audio * scalar
        # print(f'SNR: {self.compute_snr(org_audio, mixed_audio)}')
        return mixed_audio


    def forward(self, rec_audio):
        # noise_path = glob.glob(f'{self.noise_path}/*.wav')
        # id = np.random.choice(len(noise_path), 1)[0]
        # # print(f"noise choice{id}")
        # noise_audio, _ = torchaudio.load(noise_path[id])
        # noise_audio = noise_audio[:,:rec_audio.shape[-1]].to(rec_audio.device).squeeze(0)
        # # noise_audio = noise_audio / torch.max(torch.abs(noise_audio))
        # # print(noise_audio.shape)

        recap_audios = self.recapture(rec_audio).to(rec_audio.device)
        # recap_noi_audios = []
        # for recap_audio in recap_audios:
        #     out = self.add_noise(recap_audio, noise_audio, self.noise_snr)
        #     recap_noi_audios.append(out)
        # recap_noi_audios = torch.stack(recap_noi_audios, dim=0)
        # print(recap_noi_audios.shape)
        return recap_audios


if __name__ == '__main__':
    data, sr = sf.read("/raid/chenjincheng/datasets/vctk_p225_013_50cm.wav")
    data= torch.FloatTensor(data).unsqueeze(0).cuda()

    rct = RCT(10).cuda()
    data2 = rct(data)
    sf.write("/raid/chenjincheng/datasets/vctk_p225_013_50cm_recapnet_full.wav", data2.squeeze(0).cpu().numpy(), sr)
    print("finish!!")