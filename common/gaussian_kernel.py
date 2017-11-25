import numpy as np
import math

# vector m, matrix q, return mTQm
def quadratic_form(m, q):
	return np.dot(m, np.dot(q, m))

class GaussianKernel():
	def __init__(self, h, m, q):
		self.h = h
		self.m = m
		self.dimension = len(m)
		assert q.shape == (self.dimension, self.dimension)
		self.q = q

	# other: another GaussianKernel
	# returns a GaussianKernel equivalent to their product
	def multiply(self, other):
		q = self.q + other.q
		m = np.dot(np.linalg.inv(q), np.dot(self.q, self.m) + np.dot(other.q, other.m))
		h = self.h * other.h * math.exp(-0.5 * (quadratic_form(self.m, self.q) + quadratic_form(other.m, other.q) - quadratic_form(m, q)))
		return GaussianKernel(h, m, q)

	# maxing out the last num variables
	# reutrns a GaussianKernel
	def max_out(self, num):
		h = self.h
		idx = -1 * num
		m = self.m[:idx]
		q11 = self.q[:idx, :idx]
		q12 = self.q[:idx, idx:]
		q21 = self.q[idx:, :idx]
		q22 = self.q[idx:, idx:]
		q = q11 - np.dot(q12, np.dot(np.linalg.inv(q22), q21))
		return GaussianKernel(h, m, q)

	# fixes the last variables to x
	# returns a GaussianKernel
	def fix_variables(self, x):
		idx = -1 * len(x)
		q11 = self.q[:idx, :idx]

		q12 = self.q[:idx, idx:]
		q21 = self.q[idx:, :idx]
		q22 = self.q[idx:, idx:]
		q_temp = q22 - np.dot(q21, np.dot(np.linalg.inv(q11), q12))
		h = self.h * math.exp(-0.5 * quadratic_form(x - self.m[:idx], q_temp))
		m = self.m[idx:] - np.dot(np.linalg.inv(q11), np.dot(q12, x - self.m[:idx]))
		q = q11
		return GaussianKernel(h, m, q)

	# add on num variables to the end
	# returns a GaussianKernel
	def add_variables(self, num):
		h = self.h
		m = np.pad(self.m, ((0, num)), 'constant')
		q = np.pad(self.q, ((0, num), (0, num)), 'constant')
		return GaussianKernel(h, m, q)

	# returns GaussianKernel evaluated at x
	def evaluate(self, x):
		assert len(x) == len(self.m)
		return self.h * math.exp(-0.5 * quadratic_form(x - self.m, self.q))

	def __eq__(self, other):
		return self.h == other.h and self.m == other.m and self.q == other.q

	def __str__(self):
		return self.h, self.m, self.q
