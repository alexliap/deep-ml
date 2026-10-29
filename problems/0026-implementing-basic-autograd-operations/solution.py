class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		if not isinstance(other, Value):
			other = Value(data=other)
		out = self.data + other.data
		out = Value(data=out, _children=(self, other), _op="add")

		def _backward():
			self.grad += 1 * out.grad    # ∂L/∂x = ∂out/∂x * ∂L/∂out
			other.grad += 1 * out.grad   # same for y
		out._backward = _backward

		return out

	def __mul__(self, other):
		if not isinstance(other, Value):
			other = Value(data=other)
		out = self.data * other.data
		out = Value(data=out, _children=(self, other), _op="mul")

		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = _backward

		return out

	def relu(self):
		out = Value(max(0, self.data), _children=(self,), _op="relu")

		def _backward():
			if out.data > 0:
				self.grad += 1 * out.grad
			else:
				self.grad += 0

		out._backward = _backward

		return out

	def backward(self):
		topo = []
		visited = set()

		def build(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build(child)
				topo.append(v)

		build(self)

		self.grad = 1
		for node in reversed(topo):
			node._backward()