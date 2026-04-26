import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	gradient_np = np.array(gradient)
	# magnitude
	mag = np.sqrt(np.sum(gradient_np**2))
	if mag == 0:
		return {'magnitude': 0, 
	 	 'direction': [0]*len(gradient_np), 
		 'descent_direction': [0]*len(gradient_np)}
	direction = gradient_np/mag
	desc_direction = -direction

	return {'magnitude': mag, 
	'direction': direction, 
	'descent_direction': desc_direction}
