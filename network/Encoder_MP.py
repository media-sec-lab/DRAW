from . import *

class Encoder_MP(nn.Module):
	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_MP, self).__init__()
		self.H = H
		self.W = W

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)

		self.image_pre_layer = ConvBNRelu(2, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=3)
		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 2, 2, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
		message_pre_0 = self.message_pre_layer_0(message_image)
		message_pre_1 = self.message_pre_layer_1(message_pre_0)
		message_pre_2 = self.message_pre_layer_2(message_pre_1)
		intermediate2 = self.message_first_layer(message_pre_2)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)

		return output

# class Encoder_MP(nn.Module):
# 	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
# 		super(Encoder_MP, self).__init__()
# 		self.H = H
# 		self.W = W
#
# 		self.diffusion_length = diffusion_length
# 		self.diffusion_size = int(diffusion_length ** 0.5)
#
# 		self.image_pre_layer = ConvBNRelu(2, channels)
# 		self.image_first_layer = SENet(channels, channels, blocks=blocks)
#
# 		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length*2)  # 512
# 		self.message_pre_layer_0 = ConvBNRelu(1, channels)
# 		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=3)
# 		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
# 		self.message_first_layer = SENet(channels, channels, blocks=blocks)
#
# 		self.after_concat_layer = ConvBNRelu(2 * channels, channels)
#
# 		self.final_layer = nn.Conv2d(channels + 2, 2, kernel_size=1)
#
# 	def forward(self, image, message):
# 		# first Conv part of Encoder
# 		image_pre = self.image_pre_layer(image)
# 		intermediate1 = self.image_first_layer(image_pre)
#
# 		# Message Processor (with diffusion)
# 		message_duplicate = self.message_duplicate_layer(message)
# 		message_image = message_duplicate.view(-1, 1, self.diffusion_size*2, self.diffusion_size) # 32,16
# 		message_pre_0 = self.message_pre_layer_0(message_image)
# 		message_pre_1 = self.message_pre_layer_1(message_pre_0)
# 		message_pre_2 = self.message_pre_layer_2(message_pre_1)
# 		intermediate2 = self.message_first_layer(message_pre_2)
#
# 		# concatenate
# 		concat1 = torch.cat([intermediate1, intermediate2], dim=1)
#
# 		# second Conv part of Encoder
# 		intermediate3 = self.after_concat_layer(concat1)
#
# 		# skip connection
# 		concat2 = torch.cat([intermediate3, image], dim=1)
#
# 		# last Conv part of Network
# 		output = self.final_layer(concat2)
#
# 		return output


class Encoder_MP_Diffusion(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_MP_Diffusion, self).__init__()
		self.H = H
		self.W = W

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)

		self.image_pre_layer = ConvBNRelu(2, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=3)
		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 2, 2, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
		message_pre_0 = self.message_pre_layer_0(message_image)
		message_pre_1 = self.message_pre_layer_1(message_pre_0)
		message_pre_2 = self.message_pre_layer_2(message_pre_1)
		intermediate2 = self.message_first_layer(message_pre_2)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)

		return output

