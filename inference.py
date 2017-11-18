from static import *
import numpy as np
import math

Q_LAMBDA = 10.

# input
iois = []
def get_onsets_in_interval():
	ret = []
	interval_counter = 1
	onset_counter = 0
	for i in iois:
		if i <= interval_counter * INTERVAL_SIZE:
			onset_counter += 1
		else:
			ret.append(onset_counter)
			onset_counter = 1
			interval_counter += 1
	ret.append(onset_counter)
	return ret

# arr_num_onsets = get_onsets_in_interval(iois)
arr_num_onsets = [3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3]

def compute_probability_observed(num_onsets, position, rhythm_pattern):
	rhythmic_pattern_value = compute_rhythmic_pattern_value(position, rhythm_pattern)
	a = rhythmic_pattern_value**2/Q_LAMBDA
	b = rhythmic_pattern_value/Q_LAMBDA
	return (b**a*math.gamma(a + num_onsets))/(math.factorial(num_onsets)*math.gamma(a)*(b+1)**(a + num_onsets))

def compute_diagonal_matrix(num_onsets):
	num_states = len(STATES)
	mat = np.zeros((num_states, num_states))
	for i in range(num_states):
		pos, vel, met, rhy = STATES[i]
		p = compute_probability_observed(num_onsets, pos, rhy)
		mat[i][i] = p
	return mat


def compute_alpha(prev_alpha, diagonal_matrix):
	return np.dot(diagonal_matrix, np.transpose(TRANSITION_MATRIX)).dot(prev_alpha)

def compute_beta(prev_beta, diagonal_matrix):
	return np.dot(diagonal_matrix, TRANSITION_MATRIX).dot(prev_beta)

alphas = [FIRST_ALPHA]
betas = [LAST_BETA] # ordered from last to first
diagonal_matrices = []

for i in arr_num_onsets:
	d = compute_diagonal_matrix(i)
	diagonal_matrices.append(d)

num_intervals = len(arr_num_onsets)
for i in range(num_intervals - 1):
	alphas.append(compute_alpha(alphas[-1], diagonal_matrices[i]))
	betas.append(compute_beta(betas[-1], diagonal_matrices[-(i+1)]))

betas = betas[::-1]
result = []
for i in range(num_intervals):
	alpha_i = alphas[i]
	beta_i = betas[i]
	p = np.multiply(alpha_i, beta_i)
	idx = np.argmax(p)
	result.append(STATES[idx])

for r in result:
	print r


