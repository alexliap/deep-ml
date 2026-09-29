import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
	if position == 0 or d_model <= 0:
		return -1
	pos = np.arange(position)[:, np.newaxis]
	i = np.arange(d_model)[np.newaxis, :]
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
	angle_rads = pos * angle_rates
	encoding = np.zeros((position, d_model))
	encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
	encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
	return encoding.astype(np.float16)