import numpy as np
import math

# onset times called o_i in paper, seconds
onsets = []

# measure positions, S in paper, rational on interval [0, 1)
all_measure_positions = [.25, .5]

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
	return .5

# probability matrix, denoted R in paper
def get_transition_probability(current_measure_position, next_measure_position):
	return .5

# tempo, denoted T_i, measured in seconds per measure
tempo_mean = 3.
tempo_variance = 1.
# initial tempo drawn from normal
initial_tempo = np.random.normal(tempo_mean, tempo_variance**(.5))

# changes in tempo, denoted delta_n
change_variance = 1.
def get_tempo_changes(arr_measure_positions, change_variance):
	arr = []
	change = np.random.normal(0, (change_variance*get_note_length(0, arr_measure_positions[0]))**(0.5))
	arr.append(change)
	for i in range(len(arr_measure_positions) - 1):
		change = np.random.normal(0, (change_variance*get_note_length(arr_measure_positions[i], arr_measure_positions[i + 1]))**(0.5))
		arr.append(change)
	return arr

# noise in data, denoted epsilon_n
noise_variance = 1.
def get_noise_amounts(arr_measure_positions, noise_variance):
	arr = []
	noise = np.random.normal(0, (noise_variance*get_note_length(0, arr_measure_positions[0]))**(0.5))
	arr.append(noise)
	for i in range(len(arr_measure_positions) - 1):
		noise = np.random.normal(0, (noise_variance*get_note_length(arr_measure_positions[i], arr_measure_positions[i + 1]))**(0.5))
		arr.append(noise)
	return arr

# y_n = o_n - o_n-1 = l(s_n-1, s_n)t_n + epsilon_n

# n dimensional Gaussian kernel, K(x, theta)
# x, m are vectors of length n, q is a n x n matrix 
def gaussian_kernel(x, h, m, q):
	return h*math.exp(-0.5*np.dot(x - m, np.dot(q, x - m)))

# rename this later
def theta_prime(initial_measure_position, next_measure_position, initial_ioi):
	note_length = get_note_length(initial_measure_position, next_measure_position)
	# parameters for ioi given tempo
	h_1 = (2 * math.pi * note_length * noise_variance)**(-0.5)
	m_1 = np.array([0, 0])
	q_1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
	q_1 *= 1./(note_length * noise_variance)
	
	# parameters for tempo when ioi is held constant
	h_2 = h_1 * math.exp(-0.5 * initial_ioi**2 * (q_1[1][1] - q_1[1][0] * q_1[0][1] / q_1[0][0]))
	m_2 = -1 * initial_ioi * q_1[0][1] / q_1[0][0]
	q_2 = q_1[0][0]

	# parameters for tempo
	h_3 = (2 * math.pi * tempo_variance)**(-0.5)
	m_3 = tempo_mean
	q_3 = 1./tempo_variance

	# multiplying the two
	q_4 = q_2 + q_3
	m_4 = (q_2 * m_2 + q_3 * m_3)/q_4
	h_4 = h_2 * h_3 * math.exp(-0.5 * (m_2**2 * q_2 + m_3**2 * q_3 - m_4**2 * q_4))

	h_4 *= get_initial_probability(initial_measure_position) * get_transition_probability(initial_measure_position, next_measure_position)
	return h_4, m_4, q_4

def l_1(initial_measure_position, next_measure_position, initial_tempo, initial_ioi):
	return gaussian_kernel(initial_tempo, *theta_prime(initial_measure_position, next_measure_position, initial_ioi))

def theta_c(previous_measure_position, current_measure_position, current_ioi):
	note_length = get_note_length(previous_measure_position, current_measure_position)
	# parameters for ioi given tempo given current tempo
	h_1 = (2 * math.pi * note_length * noise_variance)**(-0.5)
	m_1 = np.array([0, 0])
	q_1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
	q_1 *= 1./(note_length * noise_variance)

	# parameters for current tempo when current ioi is held constant
	h_2 = h_1 * math.exp(-0.5 * current_ioi**2 * (q_1[1][1] - q_1[1][0] * q_1[0][1] / q_1[0][0]))
	m_2 = -1 * current_ioi * q_1[0][1] / q_1[0][0]
	q_2 = q_1[0][0]	

	# parameters for current tempo and previous tempo when current ioi held constant
	h_3 = h_2
	m_3 = np.array([m_2, 0])
	q_3 = np.array([[q_2, 0], [0, 0]])

	# parameters for current tempo given previous tempo
	h_4 = (2 * math.pi * change_variance * note_length)**(-0.5)
	m_4 = np.array([0, 0])
	q_4 = np.array([[1., -1.], [-1., 1.]])
	q_4 *= 1./(note_length * change_variance)

	# multiply the two
	q_5 = q_3 + q_4
	m_5 = np.dot(np.linalg.inv(q_5), (np.dot(q_3, m_3) + np.dot(q_4, m_4)))
	h_5 = h_3 * h_4 * math.exp(-0.5 * (np.dot(m_3, np.dot(q_3, m_3)) + np.dot(m_4, np.dot(q_4, m_4)) - np.dot(m_5, np.dot(q_5, m_5))) )

	h_5 *= get_transition_probability(previous_measure_position, current_measure_position)
	return h_5, m_5, q_5

def c(previous_measure_position, current_measure_position, previous_tempo, current_tempo, current_ioi):
	return gaussian_kernel(np.array([current_tempo, previous_tempo]), *theta_c(previous_measure_position, current_measure_position, current_ioi))

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
			memo[0] = [(float('-inf'), float('inf'), theta[0])]
		else:
			h_cur, m_cur, q_cur = theta[i]
			prev = memo[i - 1]
			intervals = []
			for j in range(len(prev)):
				left, right, opt = prev[j]
				h_prev, m_prev, q_prev = opt

				# solve quadratic equation for critical points
				a = q_prev - q_cur
				b = 2 * (q_cur * m_cur - q_prev * m_prev)
				c = m_prev**2 * q_prev - m_cur**2 * q_cur
				pm = math.sqrt(b**2 - 4 * a * c)
				sol1 = (-1 * b + pm)/(2. * a)
				sol2 = (-1 * b - pm)/(2. * a)

				# add any valid solutions to list of end points
				new_vals = [left, right]
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
					test_val = temp_left + 1
					if test_val > temp_right:
						test_val = (temp_left + temp_right)/2.
					new_opt = opt
					if gaussian_kernel(test_val, h_cur, m_cur, q_cur) > gaussian_kernel(test_val, h_prev, m_prev, q_prev):
						new_opt = [h_cur, m_cur, q_cur]
					new_intervals.append((temp_left, temp_right, new_opt))
				intervals += new_intervals
			memo[i] = thin_intervals(intervals)
	ret = set()
	for i in memo[-1]:
		ret.add(tuple(i[2]))
	return ret

def initial_theta(next_measure_position, initial_tempo, initial_ioi):
	arr = []
	for p in all_measure_positions:
		t = theta_prime(p, next_measure_position, initial_ioi)
		arr.append(t)
	return set(thin(arr))

def get_next_theta(old_theta, previous_measure_position, current_measure_position, current_ioi):
	arr = []
	parents = {}
	for t in old_theta:
		h_1, m_1, q_1 = t
		
		h_2 = h_1
		m_2 = np.array([m_1, 0])
		q_2 = np.array([[q_1, 0], [0, 0]])

		h_3, m_3, q_3 = theta_c(previous_measure_position, current_measure_position, current_ioi)

		# multiplying the two
		q_4 = q_2 + q_3
		m_4 = np.dot(np.linalg.inv(q_4), np.dot(q_2, m_2) + np.dot(q_3, m_3))
		h_4 = h_2 * h_3 * math.exp(-0.5 * (np.dot(m_2, np.dot(q_2, m_2)) + np.dot(m_3, np.dot(q_3, m_3)) - np.dot(m_4, np.dot(q_4, m_4))))
		
		# maxing out the first element
		h_5 = h_4
		m_5 = m_4[1]
		q_5 = q_4[1][1] - q_4[1][0] * q_4[0][1] / q_4[0][0]

		arr.append([h_5, m_5, q_5])
		parents[(h_5, m_5, q_5)] = (t, previous_measure_position)
	thinned = set(thin(arr))
	final_parents = {}
	for i in thinned:
		final_parents[i] = parents[i]
	return thinned, final_parents

iois = [.75, .75, .75, .75]

# row i and col j is theta_i(all_measure_positions[j])
theta_array = []
parent_array = []
initial_row = []
for p in range(len(all_measure_positions)):
	x = initial_theta(all_measure_positions[p], initial_tempo, iois[0])
	initial_row.append(x)
theta_array.append(initial_row)
parent_array.append([0]*len(all_measure_positions))
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
print theta_array

last_theta = set()
for x in theta_array[-1]:
	last_theta = last_theta.union(x)
sorted_theta_array = sorted(list(last_theta), key=lambda x: x[0])
best_theta = sorted_theta_array[-1]
parent = None
best_tempo = best_theta[1]
tempos = [best_tempo]
positions = []
for i in range(len(theta_array[-1])):
	if tuple(best_theta) in theta_array[-1][i]:
		positions.append(all_measure_positions[i])
		parent = parent_array[-1][i][best_theta]
		break
print positions, tempos, parent


for i in range(len(iois) - 2, 0, -1):
	best_theta = parent[0]
	positions.append(parent[1])
	parent = parent_array[i][all_measure_positions.index(parent[1])][best_theta]
positions.append(parent[1])
print positions













	

























