import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .gaussian_noise import GN
from .sample_suppression import SS, Pad
from .combined import Combined
from .lowpass_filter import LF
from .amp_scale import AS
from .time_shift import TS
from .echo_addition import EA
from .pitch_scale_modification import PSM
from .resample import RS
# from .resample import DS,US
from .time_scale_modification import TSM
from .jittering import Jitter
from .crop import Crop
from .mp3_compress import MP3
from .resample_tsm import RSTSM
from  .recap_net import RCT
from .crop_new import Crop_new, Crop_new_fix
from .other import Other