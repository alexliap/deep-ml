import numpy as np

def rnn_forward(input_sequence: list[list[float]], 
				initial_hidden_state: list[float], 
				Wx: list[list[float]], 
				Wh: list[list[float]], 
				b: list[float]) -> list[float]:
	h_t = initial_hidden_state
	for i in range(len(input_sequence)):
		ith = np.array(Wx) @ np.array(input_sequence[i])
		hth = np.array(Wh) @ np.array(h_t)
		h_t = np.tanh(ith + hth + b)
	return np.round(h_t, 4)