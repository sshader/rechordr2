import numpy as np

INTERVAL_SIZE = .02
NUM_QUANTIZED_POSITIONS = 960*4
NUM_QUANTIZED_VELOCITIES = 5
# METERS = [1] # fractions of 4/4
RHYTHM_PATTERNS = [0]
PROBABILITY_VELOCITY_CHANGE = .1
# PROBABILITY_METER_CHANGE = 0
PROBABILITY_RHYTHM_CHANGE = 0

def construct_all_states():
	x = []
	met = 1 # assuming fixed 4/4 meter
	for pos in range(NUM_QUANTIZED_POSITIONS):
		for vel in range(NUM_QUANTIZED_VELOCITIES):
			# for met in meters:
			for rhy in RHYTHM_PATTERNS:
				x.append([pos +1, vel + 1, met, rhy]) # 1 indexing
	return np.array(x)

STATES = construct_all_states()
FIRST_ALPHA = np.array([1./len(STATES)]*len(STATES))
LAST_BETA = np.array([1.]*len(STATES))

def next_position(current_pos, velocity, meter=1):
	return (current_pos + velocity - 1) % (NUM_QUANTIZED_POSITIONS * meter) + 1

def next_velocity(new_velocity, old_velocity):
		if new_velocity == old_velocity:
			return 1 - PROBABILITY_VELOCITY_CHANGE
		elif new_velocity == old_velocity - 1 or new_velocity == old_velocity + 1:
			return PROBABILITY_VELOCITY_CHANGE/2.
		else:
			return 0

# def next_meter(new_meter, old_meter, wrap):
# 	if wrap:
# 		if new_meter == old_meter:
# 			return 1 - probability_meter_change
# 		else:
# 			return probability_meter_change
# 	else:
# 		if new_meter == old_meter:
# 			return 1
# 		else:
# 			return 0


def next_rhythm_pattern(new_rhythm_pattern, old_rhythm_pattern, wrap):
	if wrap:
		if new_rhythm_pattern == old_rhythm_pattern:
			return 1 - PROBABILITY_RHYTHM_CHANGE
		else:
			return PROBABILITY_RHYTHM_CHANGE
	else:
		if new_rhythm_pattern == old_rhythm_pattern:
			return 1
		else:
			return 0

def compute_transition_probability(new_state, old_state):
	new_pos, new_vel, new_met, new_rhy = new_state
	old_pos, old_vel, old_met, old_rhy = old_state
	if new_pos != next_position(old_pos, old_vel, old_met):
		return 0
	else:
		p = 1.
		wrap = new_pos < old_pos
		p *= next_velocity(new_vel, old_vel)
		# p *= next_meter(new_met, old_met, wrap)
		p *= next_rhythm_pattern(new_rhy, old_rhy, wrap)
		return p		

def construct_transition_matrix():
	mat = []
	for i in range(len(STATES)):
		row = []
		for j in range(len(STATES)):
			p = compute_transition_probability(STATES[j], STATES[i])
			row.append(p)
		mat.append(row)
	return np.array(mat)

TRANSITION_MATRIX = construct_transition_matrix()
print TRANSITION_MATRIX

# formatted as (position from 0 to 1, peak height)
TRIPLET_SPIKES = [(0., 4), (1./6., 1), (2./6., 1), (1./2., 2), (4./6., 1), (5./6., 1), (1., 4)]
DUPLET_SPIKES = [(0., 4), (1./8., .5), (1./4., 1), (3./8., .5), (1./2., 2), (5./8., .5), (3./4., 1), (7./8., .5), (1., 4)]

BASE_HEIGHT = .1
SPIKE_WIDTH = NUM_QUANTIZED_POSITIONS/100.

def spike(width, high, low, distance_from_peak):
	slope = (high - low)/width
	return high - slope * distance_from_peak

def compute_rhythmic_pattern_value(position, rhythm_pattern):
	spikes = TRIPLET_SPIKES
	if rhythm_pattern == 0:
		spikes = DUPLET_SPIKES
	if position < 0 or position > NUM_QUANTIZED_POSITIONS:
		return 0
	for i in range(len(spikes)):
		spike_position, height = spikes[i]
		spike_position *= NUM_QUANTIZED_POSITIONS
		dist = abs(spike_position - position)
		if dist <= SPIKE_WIDTH:
			return spike(SPIKE_WIDTH, height, BASE_HEIGHT, dist)
	return BASE_HEIGHT












