import numpy as np

def calculate_brightness(img):
	if len(img) == 0:
		return -1

	length = len(img[0])
	total_bright = 0
	for i in range(len(img)):
		if len(img[i]) != length:
			return -1
		if np.any(np.array(img[i]) > 255) or np.any(np.array(img[i]) < 0):
			print("yo")
			return -1

		total_bright += np.sum(img[i])

	return total_bright/(length*len(img))
