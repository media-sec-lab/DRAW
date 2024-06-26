from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
from .pydtw import SoftDTW
from pesq import pesq,pesq_batch
from .mel_loss import MFCCLoss,CustomMSELoss
from utils.load_train_setting import *
from utils.lr_warmup import LearningRateWarmUP

class Network:

	def __init__(self, H, W, message_length, noise_layers, local_rank, batch_size, lr, double_decoder=False,
				 only_decoder=False):
		# device
		self.local_rank = local_rank
		self.lr = lr

		# network
		if not double_decoder:
			self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers)
		else:
			self.encoder_decoder = Encoder_double_Decoder(H, W, message_length, noise_layers)

		self.discriminator = Discriminator()

		# only decoder
		if only_decoder:
			for p in self.encoder_decoder.encoder.parameters():
				p.requires_grad = False

		self.encoder_decoder = torch.nn.parallel.DistributedDataParallel(self.encoder_decoder.cuda(local_rank), device_ids=[local_rank])
		# self.encoder_decoder = torch.nn.parallel.DistributedDataParallel(self.encoder_decoder.cuda(local_rank),
		# 																 device_ids=[local_rank],
		# 																 find_unused_parameters=True)
		self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator.cuda(local_rank), device_ids=[local_rank])

		if only_decoder:
			# self.encoder_decoder = torch.nn.parallel.DistributedDataParallel(self.encoder_decoder.cuda(local_rank),
			# 																 device_ids=[local_rank],
			# 																 find_unused_parameters=True)
			for p in self.encoder_decoder.module.encoder.parameters():
				p.requires_grad = False

		# mark "cover" as 1, "encoded" as 0
		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float).cuda(local_rank)
		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float).cuda(local_rank)

		# optimizer
		if opimizer_choice == "adamw":
			print("adamw")
			self.opt_encoder_decoder = torch.optim.AdamW(
				filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=self.lr)
			self.opt_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr)
		else:
			print("adam")
			self.opt_encoder_decoder = torch.optim.Adam(
				filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=self.lr)
			self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

		# schuduler
		# self.scheduler_enc_dec = torch.optim.lr_scheduler.MultiStepLR(self.opt_encoder_decoder, milestones=[90, 110], gamma=0.1, last_epoch=-1)
		self.cos_scheduler_enc_dec = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_encoder_decoder, epoch_number-warm_up_epoch)
		self.scheduler_enc_dec = LearningRateWarmUP(optimizer=self.opt_encoder_decoder,
													warmup_iteration=warm_up_epoch,
													keep_iteration=keep_epoch,
													target_lr=lr,
													after_scheduler=self.cos_scheduler_enc_dec
													)

		# loss function
		self.criterion_BCE = nn.BCEWithLogitsLoss().cuda(local_rank)
		self.criterion_MSE = nn.MSELoss().cuda(local_rank)
		self.criterion_L1 = nn.L1Loss().cuda(local_rank)
		# self.softdtw_loss = SoftDTW(gamma=1.0, normalize=True).cuda(local_rank)
		self.mfcc_loss = MFCCLoss().cuda(local_rank)
		self.weight_loss = CustomMSELoss().cuda(local_rank)

		# weight of encoder-decoder loss
		self.discriminator_weight = 0.0001
		self.encoder_weight = encoder_weight
		self.encoder_weight_audio = encoder_audio_weight
		self.decoder_weight = decoder_weight

	def train(self, spect_phase: torch.Tensor, messages: torch.Tensor, sample: torch.Tensor):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute
			# sample.shape:[B, 16130]
			spect_phase, messages, sample = spect_phase.cuda(self.local_rank), messages.cuda(self.local_rank), sample.cuda(self.local_rank)
			encoded_images, noised_spect, decoded_messages, rec_audio = self.encoder_decoder(spect_phase, messages[:,:message_length])
			# print(f"spect_phase:{spect_phase.shape}")
			'''
			train discriminator
			'''
			# sample = sample.unsqueeze(1)
			# rec_audio = rec_audio.unsqueeze(1)
			self.opt_discriminator.zero_grad()
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(spect_phase)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder = self.criterion_MSE(encoded_images, spect_phase)

			# RAW : the encoded audio should be similar to cover audio
			g_loss_on_encoder_audio = self.mfcc_loss(rec_audio, sample)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.weight_loss(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder + self.encoder_weight_audio * g_loss_on_encoder_audio

			# g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
			# 		 self.decoder_weight * g_loss_on_decoder
			g_loss.backward()
			self.opt_encoder_decoder.step()

			# snr
			# snr = self.compute_snr_batch(sample, rec_audio.detach())
			snr = 0

			# pesq
			# pesq = self.compute_pesq_batch(sample.detach(), rec_audio.detach())
			pesq = 0

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"snr": snr,
			"pesq": pesq,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_encoder_audio": g_loss_on_encoder_audio,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result


	def validation(self, spect_phase: torch.Tensor, messages: torch.Tensor, sample: torch.Tensor):
		self.encoder_decoder.eval()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			spect_phase, messages, phase = spect_phase.cuda(self.local_rank), messages.cuda(self.local_rank), sample.cuda(self.local_rank)
			encoded_images, noised_spect, decoded_messages, rec_audio = self.encoder_decoder(spect_phase, messages[:,:message_length])

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(spect_phase)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder = self.criterion_MSE(encoded_images, spect_phase)

			# RAW : the encoded audio should be similar to cover audio
			g_loss_on_encoder_audio = self.mfcc_loss(rec_audio, sample)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.weight_loss(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder + self.encoder_weight_audio * g_loss_on_encoder_audio

			# g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
			# 		 self.decoder_weight * g_loss_on_decoder

			# snr
			snr = self.compute_snr_batch(sample, rec_audio.detach())

			# pesq
			pesq = self.compute_pesq_batch(sample.detach(), rec_audio.detach())
			# pesq = 0
		#

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"snr": snr,
			"pesq": pesq,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_encoder_audio": g_loss_on_encoder_audio,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		# self.scheduler_enc_dec.step()

		return result, (spect_phase, encoded_images, noised_spect, messages, decoded_messages)

	def train_only_decoder(self, spect_phase: torch.Tensor, messages: torch.Tensor, sample: torch.Tensor):
		self.encoder_decoder.train()

		with torch.enable_grad():
			# use device to compute
			spect_phase, messages, sample = spect_phase.cuda(self.local_rank), messages.cuda(
				self.local_rank), sample.cuda(self.local_rank)
			encoded_images, noised_spect, decoded_messages, rec_audio = self.encoder_decoder(spect_phase, messages)

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# RESULT : the decoded message should be similar to the raw message
			g_loss = self.criterion_MSE(decoded_messages, messages)

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# snr
			snr = self.compute_snr_batch(sample, rec_audio.detach())

			# pesq
			# pesq = self.compute_pesq_batch(sample.detach(), rec_audio.detach())
			pesq = 0

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"snr": snr,
			"pesq": pesq,
			"g_loss": g_loss,
			"g_loss_on_discriminator": 0.,
			"g_loss_on_encoder": 0.,
			"g_loss_on_encoder_audio": 0.,
			"g_loss_on_decoder": 0.,
			"d_cover_loss": 0.,
			"d_encoded_loss": 0.,
		}
		return result

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		error_rate /= batch_size
		return error_rate

	def compute_snr(self, input_signal, output_signal):
		Ps = torch.sum(torch.abs(input_signal ** 2))
		Pn = torch.sum(torch.abs((input_signal - output_signal) ** 2))
		return 10 * torch.log10((Ps / Pn))

	def compute_snr_batch(self, input_signals, output_signals):
		snr = 0.0
		batch_size = len(input_signals)
		for i in range(batch_size):
			snr += self.compute_snr(input_signals[i], output_signals[i])
		snr /= batch_size
		return snr

	def compute_pesq(self, input_signal, output_signal):
		input_signal = input_signal.cpu().numpy()
		output_signal = output_signal.cpu().numpy()
		return pesq(16000, input_signal, output_signal)

	def compute_pesq_batch(self, input_signals, output_signals):
		pesq = 0.0
		batch_size = len(input_signals)
		for i in range(batch_size):
			pesq += self.compute_pesq(input_signals[i], output_signals[i])
		pesq /= batch_size
		return pesq

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder, map_location=torch.device("cuda")))

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator, map_location=torch.device("cuda")))

