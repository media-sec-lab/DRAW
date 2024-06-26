import torch
import torch.nn as nn


class EA(nn.Module):
    """
    Echo_addition noise layer. Add a echo signal to original signal
    """

    def __init__(self):
        super(EA, self).__init__()
        self.sample_rate = 16000
        self.delay_time = 0.1  # second
        self.decay_strength = 0.30

    def echo_addition(self, audio):
        delay_length = int(self.sample_rate * self.delay_time)
        dacay_signal = audio * self.decay_strength  # [batch_size, num_sample]

        echo_signal = torch.cat(
            (dacay_signal[:, -delay_length:], dacay_signal[:, :-delay_length]), dim=-1
        )

        return audio + echo_signal

    def forward(self, rec_audio):
        noised_audio = self.echo_addition(rec_audio)
        return noised_audio
