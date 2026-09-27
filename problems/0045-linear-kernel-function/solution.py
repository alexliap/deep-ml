import numpy as np

def kernel_function(x1, x2):
	out = 0
	for i in range(len(x1)):
		out += x1[i]*x2[i]
	
	return out
