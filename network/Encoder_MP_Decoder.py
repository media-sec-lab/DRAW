from . import *
from .Encoder_MP import Encoder_MP
from .Decoder import Decoder
from .Noise import Noise
import torch


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

class EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder, self).__init__()
		self.encoder = Encoder_MP(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder(H, W, message_length)
		self.noise_layer = noise_layers
		self.frame_length = 16129
	def forward(self, spect_phase_frame, message):
		# sp:[B,frame, 2, 128, 128]
		encoded_sp_list = []
		rec_audio = []
		decoded_message = []
		# print(f"{self.noise.noise.__class__.__name__}")
		# print(f"{self.noise_layer.list[id]}")
		for i in range(spect_phase_frame.shape[1]):
			spect_phase = spect_phase_frame[:,i,...]
			# print(f"spect_phase:{spect_phase.shape}")
			encoded_sp_frame = self.encoder(spect_phase, message)   # [B,2, 128, 128]
			encoded_sp_frame_1 = encoded_sp_frame.permute(0, 2, 3, 1)
			rec_audio_frame = torch.istft(encoded_sp_frame_1, 254, 127, length=16129)  # [B, 16129]

			encoded_sp_list.append(encoded_sp_frame)
			rec_audio.append(rec_audio_frame)
		encoded_sp = torch.stack(encoded_sp_list, dim=1)  #[B,frame, 2, 128, 128]
		rec_audio = torch.cat(rec_audio,dim=1)  # [B, 161290]
		# print(f"encoded_sp:{encoded_sp.shape}")
		# print(f"rec_audio:{rec_audio.shape}")

		noised_audio = self.noise(rec_audio)
		# noised_audio_list = audio_split(noised_audio)

		# print(f'noised_audio:{noised_audio.shape}')
		# original attack

		# crop
		if isinstance(noised_audio,list):
			# print("original crop")
			crop_audio_list = noised_audio[0]
			pad_audio_list = noised_audio[1]
			for l in range(len(crop_audio_list)):
				# print(f"noise:{crop_audio_list[l].shape}")
				# print(f"syn:{pad_audio_list[l].shape}")

				syncode_spect_phase = torch.view_as_real(torch.stft(pad_audio_list[l], 254, 127, return_complex=True))
				syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
				decoded_syncode_frame = self.decoder(syncode_spect_phase)[:, :32]

				noised_spect_phase = torch.view_as_real(torch.stft(crop_audio_list[l], 254, 127, return_complex=True))
				noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
				decoded_message_frame = self.decoder(noised_spect_phase)[:, 32:]

				decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)
				decoded_message.append(decoded_message_frame)
		elif isinstance(noised_audio, tuple):
			if noised_audio[0] == "jitter":
				# print("jitter")
				start_decode_sample = self.frame_length-self.frame_length//100
				# print(start_decode_sample)
				noised_audio_list = audio_split(noised_audio[1].to(message.device))
				noised_audio = noised_audio[1]
				# print(f"noise_audio:{noised_audio.shape}")
				for k in range(len(noised_audio_list)):
					# print(f"noise:{noised_audio_list[k].shape}")
					noised_spect_phase = torch.view_as_real(
						torch.stft(noised_audio_list[k], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.decoder(noised_spect_phase)[:,32:]

					if k < len(noised_audio_list)-1 :
						# print(f"syn: decode {k}:{start_decode_sample * k, start_decode_sample * k + self.frame_length}")

						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:start_decode_sample*k+self.frame_length], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.decoder(syncode_spect_phase)[:, :32]

					else:
						# print(f"syn: decode {k}:{start_decode_sample * k, noised_audio.shape[-1]}")
						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.decoder(syncode_spect_phase)[:, :32]
					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)

					decoded_message.append(decoded_message_frame)

			else:
				# print("tsm")

				tsm_rate = noised_audio[0]
				start_decode_sample = int(tsm_rate*self.frame_length)
				noised_audio_list = audio_split(noised_audio[1].to(message.device))
				noised_audio = noised_audio[1]
				# print(f"noise_audio:{noised_audio.shape}")
				for k in range(len(noised_audio_list)):
					# print(f"syn: decode {k}:{start_decode_sample*k, start_decode_sample*k+self.frame_length}")
					# print(f"noise: {noised_audio_list[k].shape}")
					noised_spect_phase = torch.view_as_real(
						torch.stft(noised_audio_list[k], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.decoder(noised_spect_phase)[:,32:]


					if k < len(noised_audio_list)-1 :
						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:start_decode_sample*k+self.frame_length], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.decoder(syncode_spect_phase)[:, :32]

					else:
						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.decoder(syncode_spect_phase)[:, :32]
					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)
					decoded_message.append(decoded_message_frame)
						# print("decode last!")
		else:
			# print("others")
			noised_audio_list = audio_split(noised_audio.to(message.device))
			for j in range(len(noised_audio_list)):
				# print(noised_audio_list[j].shape)
				noised_spect_phase = torch.view_as_real(torch.stft(noised_audio_list[j], 254, 127, return_complex=True))
				noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
				decoded_message_frame = self.decoder(noised_spect_phase)
				decoded_message.append(decoded_message_frame)

		decoded_message = torch.cat(decoded_message,dim=1)
		return encoded_sp, noised_spect_phase, decoded_message, rec_audio

class Encoder_double_Decoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(Encoder_double_Decoder, self).__init__()
		self.encoder = Encoder_MP(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.syn_decoder = Decoder(H, W, message_length)
		self.watermark_decoder = Decoder(H, W, message_length)
		self.noise_layer = noise_layers
		self.frame_length = 16129
	def forward(self, spect_phase_frame, message):
		# sp:[B,frame, 2, 128, 128]
		encoded_sp_list = []
		rec_audio = []
		decoded_message = []
		# print(f"{self.noise.noise.__class__.__name__}")
		# print(f"{self.noise_layer.list[id]}")
		for i in range(spect_phase_frame.shape[1]):
			spect_phase = spect_phase_frame[:,i,...]
			# print(f"spect_phase:{spect_phase.shape}")
			encoded_sp_frame = self.encoder(spect_phase, message)   # [B,2, 128, 128]
			encoded_sp_frame_1 = encoded_sp_frame.permute(0, 2, 3, 1)
			rec_audio_frame = torch.istft(encoded_sp_frame_1, 254, 127, length=16129)  # [B, 16129]

			encoded_sp_list.append(encoded_sp_frame)
			rec_audio.append(rec_audio_frame)
		encoded_sp = torch.stack(encoded_sp_list, dim=1)  #[B,frame, 2, 128, 128]
		rec_audio = torch.cat(rec_audio,dim=1)  # [B, 161290]
		# print(f"encoded_sp:{encoded_sp.shape}")
		# print(f"rec_audio:{rec_audio.shape}")

		noised_audio = self.noise(rec_audio)
		# noised_audio_list = audio_split(noised_audio)

		# print(f'noised_audio:{noised_audio.shape}')
		# original attack

		# crop
		if isinstance(noised_audio,list):

			if noised_audio[-1] == "crop":
				# print("original crop")
				crop_audio_list = noised_audio[0]
				pad_audio_list = noised_audio[1]
				for l in range(len(crop_audio_list)):
					# print(f"noise:{crop_audio_list[l].shape}")
					# print(f"syn:{pad_audio_list[l].shape}")

					syncode_spect_phase = torch.view_as_real(torch.stft(pad_audio_list[l], 254, 127, return_complex=True))
					syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
					decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]

					noised_spect_phase = torch.view_as_real(torch.stft(crop_audio_list[l], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.watermark_decoder(noised_spect_phase)[:, 31:]

					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)
					decoded_message.append(decoded_message_frame)
			elif noised_audio[-1] == "pad":
				# print("Padding")
				ori_audio_list = noised_audio[0]
				pad_audio_list = noised_audio[1]
				for m in range(len(ori_audio_list)):
					# print(f"noise:{crop_audio_list[l].shape}")
					# print(f"syn:{pad_audio_list[l].shape}")

					syncode_spect_phase = torch.view_as_real(torch.stft(ori_audio_list[m], 254, 127, return_complex=True))
					syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
					decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]

					noised_spect_phase = torch.view_as_real(torch.stft(pad_audio_list[m], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.watermark_decoder(noised_spect_phase)[:, 31:]

					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)
					decoded_message.append(decoded_message_frame)
		elif isinstance(noised_audio, tuple):
			if noised_audio[0] == "jitter":
				# print("jitter")
				start_decode_sample = self.frame_length-self.frame_length//100
				# print(start_decode_sample)
				noised_audio_list = audio_split(noised_audio[1].to(message.device))
				noised_audio = noised_audio[1]
				# print(f"noise_audio:{noised_audio.shape}")
				for k in range(len(noised_audio_list)):
					# print(f"noise:{noised_audio_list[k].shape}")

					noised_spect_phase = torch.view_as_real(
						torch.stft(noised_audio_list[k], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.watermark_decoder(noised_spect_phase)[:,31:]

					if k < len(noised_audio_list)-1 :
						# print(f"syn: decode {k}:{start_decode_sample * k, start_decode_sample * k + self.frame_length}")

						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:start_decode_sample*k+self.frame_length], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]

					else:
						# print(f"syn jitter : decode {k}:{noised_audio[:,-16129:].shape}")
						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,-16129:], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]

					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)

					decoded_message.append(decoded_message_frame)

			else:
				# print("tsm")

				tsm_rate = noised_audio[0]
				start_decode_sample = int(tsm_rate*self.frame_length)
				noised_audio_list = audio_split(noised_audio[1].to(message.device))
				noised_audio = noised_audio[1]
				# print(f"noise_audio:{noised_audio.shape}")
				for k in range(len(noised_audio_list)):
					# print(f"syn: decode {k}:{start_decode_sample*k, start_decode_sample*k+self.frame_length}")
					# print(f"noise: {noised_audio_list[k].shape}")

					noised_spect_phase = torch.view_as_real(
						torch.stft(noised_audio_list[k], 254, 127, return_complex=True))
					noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
					decoded_message_frame = self.watermark_decoder(noised_spect_phase)[:,31:]

					if k < len(noised_audio_list)-1 :
						syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:start_decode_sample*k+self.frame_length], 254, 127, return_complex=True))
						syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
						decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]

					else:
						if tsm_rate <= 1:
							# print(f"syn tsm {tsm_rate}: decode {k}:{noised_audio[:, -16129:].shape}")

							syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,-16129:], 254, 127, return_complex=True))
							syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
							decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]
						else:
							# print(f"syn tsm {tsm_rate}: decode {k}:{start_decode_sample*k, start_decode_sample*k+self.frame_length}")

							syncode_spect_phase = torch.view_as_real(torch.stft(noised_audio[:,start_decode_sample*k:start_decode_sample*k+self.frame_length], 254, 127, return_complex=True))
							syncode_spect_phase = syncode_spect_phase.permute(0, 3, 1, 2)
							decoded_syncode_frame = self.syn_decoder(syncode_spect_phase)[:, :31]


					decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)
					decoded_message.append(decoded_message_frame)
						# print("decode last!")
		else:
			# print("others")
			noised_audio_list = audio_split(noised_audio.to(message.device))
			for j in range(len(noised_audio_list)):
				# print(f"{noised_audio_list[j].shape}")
				noised_spect_phase = torch.view_as_real(torch.stft(noised_audio_list[j], 254, 127, return_complex=True))
				noised_spect_phase = noised_spect_phase.permute(0, 3, 1, 2)
				decoded_syncode_frame = self.syn_decoder(noised_spect_phase)[:, :31]
				decoded_message_frame = self.watermark_decoder(noised_spect_phase)[:,31:]

				decoded_message_frame = torch.cat((decoded_syncode_frame, decoded_message_frame), dim=1)

				decoded_message.append(decoded_message_frame)

		decoded_message = torch.cat(decoded_message,dim=1)
		return encoded_sp, noised_spect_phase, decoded_message, rec_audio

