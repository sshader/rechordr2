import numpy as np
import math

import sys
sys.path.append('..')
from common.gaussian_kernel import *

# onset times called o_i in paper, seconds
onsets = [0, 0.453642, 0.935058, 1.823826, 2.245065, 2.657046, 3.448605,
					3.865215, 4.346631, 5.175222, 5.577945, 6.0177, 6.874065, 7.281417,
					7.744317, 8.623827, 9.008034, 9.466305, 10.304154, 10.706877,
					11.146632, 11.98911, 12.414978, 12.86862, 13.262085, 13.706469,
					14.136966, 15.687681, 17.275428, 17.747586, 17.803134, 18.659499,
					19.11777, 19.53438, 20.409261, 20.816613, 21.251739, 22.640439,
					24.066171, 24.279105, 24.542958, 24.973455, 25.394694, 25.834449]

iois = []
for i in range(len(onsets) - 1):
	iois.append(onsets[i+1] - onsets[i])

# measure positions, S in paper, rational on interval [0, 1)
all_measure_positions = [0, .125, .25, .375, .5, .625, .75, .875]

# l(s_n, s_n+1) in paper
def get_note_length(current_measure_position, next_measure_position):
	if current_measure_position < next_measure_position:
		return next_measure_position - current_measure_position
	else:
		return 1 + next_measure_position - current_measure_position

# score position, called m_n in paper
def get_score_positions(arr_measure_positions):
	arr = []
	arr.append(get_note_length(0, arr_measure_positions[0]))
	for i in range(len(arr_measure_positions) - 1):
		arr.append(get_note_length(arr_measure_positions[i], arr_measure_positions[i + 1]))
	return arr

# initial distribution, denoted I(s_0) in paper
def get_initial_probability(measure_position):
	return .125

# probability matrix, denoted R in paper
def get_transition_probability(current_measure_position, next_measure_position):
	return .125

# tempo, denoted T_i, measured in seconds per measure
tempo_mean = 1.6
tempo_variance = .2
# initial tempo drawn from normal
def get_initial_tempo(tempo_mean, tempo_variance):
	return np.random.normal(tempo_mean, tempo_variance**(.5))

# changes in tempo, denoted delta_n
# is maybe unnecessary
change_variance = .1
def get_tempo_changes(arr_measure_positions, change_variance):
	arr = []
	change = np.random.normal(0, (change_variance*get_note_length(0, arr_measure_positions[0]))**(0.5))
	arr.append(change)
	for i in range(len(arr_measure_positions) - 1):
		change = np.random.normal(0, (change_variance*get_note_length(arr_measure_positions[i], arr_measure_positions[i + 1]))**(0.5))
		arr.append(change)
	return arr

# noise in data, denoted epsilon_n
# is maybe unnecessary
noise_variance = .1
def get_noise_amounts(arr_measure_positions, noise_variance):
	arr = []
	noise = np.random.normal(0, (noise_variance*get_note_length(0, arr_measure_positions[0]))**(0.5))
	arr.append(noise)
	for i in range(len(arr_measure_positions) - 1):
		noise = np.random.normal(0, (noise_variance*get_note_length(arr_measure_positions[i], arr_measure_positions[i + 1]))**(0.5))
		arr.append(noise)
	return arr

# y_n = o_n - o_n-1 = l(s_n-1, s_n)t_n + epsilon_n

# rename this later
def theta_prime(initial_measure_position, next_measure_position, initial_ioi):
	note_length = get_note_length(initial_measure_position, next_measure_position)
	# parameters for ioi given tempo
	h1 = (2 * math.pi * note_length * noise_variance)**(-0.5)
	m1 = np.array([0, 0])
	q1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
	q1 *= 1./(note_length * noise_variance)
	ioi_given_tempo_kernel = GaussianKernel(h1, m1, q1)
	
	# parameters for tempo when ioi is held constant
	tempo_fixed_ioi_kernel = ioi_given_tempo_kernel.fix_variables([initial_ioi])

	# parameters for tempo
	h2 = (2 * math.pi * tempo_variance)**(-0.5)
	m2 = tempo_mean
	q2 = 1./tempo_variance
	tempo_kernel = GaussianKernel(h2, m2, q2)

	# multiplying the two
	final_kernel = tempo_fixed_ioi_kernel.multiply(tempo_kernel)
	final_kernel.h *= get_initial_probability(initial_measure_position) * get_transition_probability(initial_measure_position, next_measure_position)
	return final_kernel

# maybe not needed
def likelihood(initial_measure_position, next_measure_position, initial_tempo, initial_ioi):
	kernel = theta_prime(initial_measure_position, next_measure_position, initial_ioi)
	return kernel.evaluate(np.array([initial_ioi]))

def theta_c(previous_measure_position, current_measure_position, current_ioi):
	note_length = get_note_length(previous_measure_position, current_measure_position)
	# parameters for ioi given tempo given current tempo
	h1 = (2 * math.pi * note_length * noise_variance)**(-0.5)
	m1 = np.array([0, 0])
	q1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
	q1 *= 1./(note_length * noise_variance)
	ioi_given_tempo_kernel = GaussianKernel(h1, m1, q1)

	# parameters for current tempo when ioi is held constant
	tempo_fixed_ioi_kernel = ioi_given_tempo_kernel.fix_variables([current_ioi])

	# parameters for current tempo and previous tempo when current ioi held constant
	tempos_fixed_ioi_kernel = tempo_fixed_ioi_kernel.add_variables(1)

	# parameters for current tempo given previous tempo
	h2 = (2 * math.pi * change_variance * note_length)**(-0.5)
	m2 = np.array([0, 0])
	q2 = np.array([[1., -1.], [-1., 1.]])
	q2 *= 1./(note_length * change_variance)
	tempos_kernel = GaussianKernel(h2, m2, q2)

	# multiply the two
	final_kernel = tempos_fixed_ioi_kernel.multiply(tempos_kernel)

	final_kernel.h *= get_transition_probability(previous_measure_position, current_measure_position)
	return final_kernel

# maybe not necessary?
def c(previous_measure_position, current_measure_position, previous_tempo, current_tempo, current_ioi):
	kernel = theta_c(previous_measure_position, current_measure_position, current_ioi)
	return kernel.evaluate(np.array([current_tempo, previous_tempo]))


tempo_min = 0.
tempo_max = 2.4
def thin_intervals(intervals):
	final_intervals = []
	left, right, opt = intervals[0]
	for i in range(1, len(intervals)):
		temp_left, temp_right, temp_opt = intervals[i]
		if temp_opt == opt:
			right = temp_right
		else:
			final_intervals.append((left, right, opt))
			left = temp_left
			right = temp_right
			opt = temp_opt
	final_intervals.append((left, right, opt))
	return final_intervals

def thin(theta):
	memo = [0]*len(theta)
	# formatted as (left endpoint, right endpoint, optimal theta)
	for i in range(len(theta)):
		if i == 0:
			memo[0] = [(tempo_min, tempo_max, GaussianKernel(*theta[0]))]
		else:
			last = theta[i]
			last = GaussianKernel(*last)
			prev = memo[i - 1]
			intervals = []
			for j in range(len(prev)):
				left, right, opt = prev[j]

				# solve quadratic equation for critical points
				a = float(opt.q - last.q)
				b = 2 * float((last.q * last.m - opt.q * opt.m))
				c = float(opt.m**2 * opt.q - last.m**2 * last.q)
				new_vals = [left, right]
				if b**2 - 4 * a * c >= 0 and a != 0:
					pm = math.sqrt(b**2 - 4 * a * c)
					sol1 = (-1 * b + pm)/(2. * a)
					sol2 = (-1 * b - pm)/(2. * a)

					# add any valid solutions to list of end points
					if left < sol1 and sol1 < right:
						new_vals.append(sol1)
					if left < sol2 and sol2 < right:
						new_vals.append(sol2)
					new_vals.sort()

				# go through each interval and use a test value
				new_intervals = []
				for k in range(len(new_vals) - 1):
					temp_left = new_vals[k]
					temp_right = new_vals[k + 1]
					test_val = (temp_left + temp_right)/2.
					new_opt = opt
					if last.evaluate(np.array([test_val])) > opt.evaluate(np.array([test_val])):
						new_opt = last
					new_intervals.append((temp_left, temp_right, new_opt))
				intervals += new_intervals
			memo[i] = thin_intervals(intervals)
	ret = []
	for i in memo[-1]:
		ret.append(i[2].convert_to_tuple())
	return ret

def initial_theta(next_measure_position, initial_tempo, initial_ioi):
	arr = []
	parents = {}
	for p in all_measure_positions:
		t = theta_prime(p, next_measure_position, initial_ioi)
		arr.append(t.convert_to_tuple())
		parents[t.convert_to_tuple()] = p
	thinned = thin(arr)
	final_parents = {}
	for i in thinned:
		final_parents[i] = parents[i]
	return thinned, final_parents


def get_next_theta(old_theta, previous_measure_position, current_measure_position, current_ioi):
	arr = []
	parents = {}
	for t in old_theta:
		kernel_t = GaussianKernel(*t).add_variables(1)

		kernel_c = theta_c(previous_measure_position, current_measure_position, current_ioi)

		# multiplying the two
		product_kernel = kernel_t.multiply(kernel_c)
		
		# maxing out the first element
		product_kernel.m = product_kernel.m[::-1]
		product_kernel.q = np.fliplr(np.flipud(product_kernel.q))
		
		final_kernel = product_kernel.max_out(1)
		# this is awful, but since I know these are 1-D I will store them
		# as a tuple (h, m, q) so it's hashable
		final_kernel_tuple = tuple([final_kernel.h, float(final_kernel.m), float(final_kernel.q)])
		# t_tuple = tuple([t.h, float(t.m), float(t.q)])
		arr.append(final_kernel_tuple)
		# gotta make this hashable
		parents[final_kernel_tuple] = (t, previous_measure_position)
	thinned = thin(arr)
	final_parents = {}
	for i in thinned:
		final_parents[i] = parents[i]
	return thinned, final_parents

# iois = [484.83265306122394, 468.11428571428587, 986.3836734693887,
# 				462.54149659863924, 445.82312925169936, 964.092517006804,
# 				518.269387755101, 445.823129251703, 991.9564625850326, 
# 				479.2598639455791, 490.40544217687057, 930.6557823129242]


# row i and col j is theta_i(all_measure_positions[j])
theta_array = []
parent_array = []
initial_row = []
parent_row = []
for p in range(len(all_measure_positions)):
	initial_tempo = get_initial_tempo(tempo_mean, tempo_variance)
	x, parents = initial_theta(all_measure_positions[p], initial_tempo, iois[0])
	initial_row.append(x)
	parent_row.append(parents)
theta_array.append(initial_row)
parent_array.append(parent_row)
for i in range(1, len(iois)):
	last_row = theta_array[-1]
	cur_row = []
	parent_row = []
	for p1 in range(len(all_measure_positions)):
		thetas = set()
		parents = {}
		for p2 in range(len(all_measure_positions)):
			x, parents_x = get_next_theta(last_row[p2], all_measure_positions[p2], all_measure_positions[p1], iois[i])
			thetas = thetas.union(x)
			parents.update(parents_x)
		cur_row.append(thetas)
		parent_row.append(parents)
	theta_array.append(cur_row)
	parent_array.append(parent_row)

last_theta = set()
for x in theta_array[-1]:
	last_theta = last_theta.union(x)
sorted_theta_array = sorted(list(last_theta), key=lambda x: x[0])
best_theta = sorted_theta_array[-1]
parent = None
best_tempo = float(best_theta[1])
tempos = [best_tempo]
positions = []
for i in range(len(theta_array[-1])):
	if best_theta in theta_array[-1][i]:
		positions.append(all_measure_positions[i])
		parent = parent_array[-1][i][best_theta]
		break
# print positions, tempos, parent


for i in range(len(iois) - 2, -1, -1):
	best_theta = parent[0]
	positions.append(parent[1])
	parent = parent_array[i][all_measure_positions.index(parent[1])][best_theta]
positions.append(parent)
print iois
print positions[::-1]
