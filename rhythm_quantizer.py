import numpy as np
import math

import sys
sys.path.append('..')
from common.gaussian_kernel import *

class RhythmQuantizer():
		def __init__(self, 
			all_measure_positions, 
			initial_probability, 
			transition_probability, 
			tempo_mean, 
			tempo_variance, 
			tempo_change_variance, 
			tempo_noise_variance,
			tempo_min,
			tempo_max):
				self.all_measure_positions = sorted(all_measure_positions)
				self.initial_probability = initial_probability
				self.transition_probability = transition_probability
				self.tempo_mean = tempo_mean
				self.tempo_variance = tempo_variance
				self.tempo_change_variance = tempo_change_variance
				self.tempo_noise_variance = tempo_noise_variance
				self.tempo_min = tempo_min
				self.tempo_max = tempo_max
				
		def quantize(self, iois):
			kernels = self._compute_kernels(iois)
			optimum_kernel = self._find_optimum_kernel(kernels)
			positions = self._traverse_parents(optimum_kernel, kernels)
			return positions

		def _traverse_parents(self, optimum_kernel, kernels):
			leaf = None
			positions = []
			# find first leaf
			for i in range(len(kernels[-1])):
				d = kernels[-1][i]
				if optimum_kernel in d.keys():
					leaf = d[optimum_kernel]
					positions.append(self.all_measure_positions[i])
					# append parent position
					positions.append(leaf[0])
			# find all other leaves
			for j in range(len(kernels) - 2, 0, -1):
				leaf = self._find_parent(leaf, j, kernels)
				positions.append(leaf[0])
			return positions[::-1]

		def _find_parent(self, leaf, index, kernels):
			print leaf
			parent_position, parent = leaf
			d = kernels[index][self.all_measure_positions.index(parent_position)]
			print d
			return d[parent]

		def _find_optimum_kernel(self, kernels):
			last_row = set()
			for x in kernels[-1]:
				last_row = last_row.union(x.keys())
			sorted_kernels = sorted(list(last_row), key = lambda x: x[0])
			optimum_kernel = sorted_kernels[-1]
			# if we care about tempos it would happen here
			return optimum_kernel	

		# nested arrays:
		# onset index: array
		#		measure position index: dictionary of kernels
		# 		kernel: measure position of parent, parent tuple
		def _compute_kernels(self, iois):
			onsets_to_positions_arr = []
			onsets_to_positions_arr.append(self._compute_first_row(iois[0]))
			for i in range(len(iois)):
				onsets_to_positions_arr.append(self._compute_next_row(onsets_to_positions_arr[-1], iois[i]))
			return onsets_to_positions_arr

		def _compute_first_row(self, ioi):
			initial_tempo = self._get_initial_tempo()
			initial_positions_to_dicts_arr = []
			for p in self.all_measure_positions:
				initial_positions_to_dicts_arr.append(self._get_initial_kernels(p, initial_tempo, ioi))
			return initial_positions_to_dicts_arr

		def _compute_next_row(self, last_row, current_ioi):
			positions_to_dicts_arr = []
			for idx in range(len(self.all_measure_positions)):
				positions_to_dicts_arr.append(self._get_kernels(
					last_row[idx].keys(), self.all_measure_positions[idx], current_ioi))
			return positions_to_dicts_arr
		
		def _get_initial_kernels(self, next_measure_position, initial_tempo, initial_ioi):
			kernel_and_parents = {}
			for p in self.all_measure_positions:
				kernel = self._initial_kernel(p, next_measure_position, initial_ioi)
				# kernel: (current_measure position, None)
				kernel_and_parents[kernel.convert_to_tuple()] = (p, None)
			thinned = self._thin(list(kernel_and_parents.keys()))
			return { key: kernel_and_parents[key] for key in thinned }

		def _get_kernels(
			self, 
			old_kernels, 
			current_measure_position, 
			current_ioi):
			kernel_and_parents = {}
			for old_kernel in old_kernels:
				old_kernel = GaussianKernel(*old_kernel)
				old_kernel_with_variable = old_kernel.add_variables(1)
				for p in self.all_measure_positions:

					new_kernel = self._kernel(p, current_measure_position, current_ioi)

					product_kernel = old_kernel_with_variable.multiply(new_kernel)

					# maxing out the first element
					product_kernel.m = product_kernel.m[::-1]
					product_kernel.q = np.fliplr(np.flipud(product_kernel.q))
					
					final_kernel = product_kernel.max_out(1)
					# kernel: (measure position of parent, parent)
					kernel_and_parents[final_kernel.convert_to_tuple()] = (p, old_kernel.convert_to_tuple())
			thinned = self._thin(list(kernel_and_parents.keys()))
			return { key: kernel_and_parents[key] for key in thinned }

		# modify to have note lengths > 1
		def _get_note_length(self, current_measure_position, next_measure_position):
				if current_measure_position < next_measure_position:
						return next_measure_position - current_measure_position
				else:
						return 1 + next_measure_position - current_measure_position
				
		def _get_initial_tempo(self):
				return np.random.normal(self.tempo_mean, self.tempo_variance**(.5))
		
		def _initial_kernel(self, initial_measure_position, next_measure_position, initial_ioi):
			note_length = self._get_note_length(initial_measure_position, next_measure_position)
			# parameters for ioi given tempo
			h1 = (2 * math.pi * note_length * self.tempo_noise_variance)**(-0.5)
			m1 = np.array([0, 0])
			q1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
			q1 *= 1./(note_length * self.tempo_noise_variance)
			ioi_given_tempo_kernel = GaussianKernel(h1, m1, q1)

			# parameters for tempo when ioi is held constant
			tempo_fixed_ioi_kernel = ioi_given_tempo_kernel.fix_variables([initial_ioi])

			# parameters for tempo
			h2 = (2 * math.pi * self.tempo_variance)**(-0.5)
			m2 = self.tempo_mean
			q2 = 1./self.tempo_variance
			tempo_kernel = GaussianKernel(h2, m2, q2)

			# multiplying the two
			final_kernel = tempo_fixed_ioi_kernel.multiply(tempo_kernel)
			final_kernel.h *= self.initial_probability(initial_measure_position) * self.transition_probability(initial_measure_position, next_measure_position)
			return final_kernel
		
		def _kernel(self, previous_measure_position, current_measure_position, current_ioi):
			note_length = self._get_note_length(previous_measure_position, current_measure_position)
			# parameters for ioi given tempo given current tempo
			h1 = (2 * math.pi * note_length * self.tempo_noise_variance)**(-0.5)
			m1 = np.array([0, 0])
			q1 =  np.array([[note_length**2, -1. * note_length], [-1. * note_length, 1.]])
			q1 *= 1./(note_length * self.tempo_noise_variance)
			ioi_given_tempo_kernel = GaussianKernel(h1, m1, q1)

			# parameters for current tempo when ioi is held constant
			tempo_fixed_ioi_kernel = ioi_given_tempo_kernel.fix_variables([current_ioi])

			# parameters for current tempo and previous tempo when current ioi held constant
			tempos_fixed_ioi_kernel = tempo_fixed_ioi_kernel.add_variables(1)

			# parameters for current tempo given previous tempo
			h2 = (2 * math.pi * self.tempo_change_variance * note_length)**(-0.5)
			m2 = np.array([0, 0])
			q2 = np.array([[1., -1.], [-1., 1.]])
			q2 *= 1./(note_length * self.tempo_change_variance)
			tempos_kernel = GaussianKernel(h2, m2, q2)

			# multiply the two
			final_kernel = tempos_fixed_ioi_kernel.multiply(tempos_kernel)

			final_kernel.h *= self.transition_probability(previous_measure_position, current_measure_position)
			return final_kernel

		# intervals formated as list of tuples (left, right, optimum)
		# collapse into intervals where no two adjacent have the same optimum
		def _collapse_intervals(self, intervals):
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

		# recomputes thinned intervals when adding the kernel
		def _add_and_thin(self, intervals, kernel):
			for i in intervals:
				new_intervals = []
				new_intervals += self._get_new_intervals(i, kernel)
			return new_intervals

		def _thin(self, kernels):
			# formatted as (left endpoint, right endpoint, optimal theta)
			intervals = [(self.tempo_min, self.tempo_max, GaussianKernel(*kernels[0]))]

			for i in range(1, len(kernels)):
				kernel = kernels[i]
				kernel = GaussianKernel(*kernel)
				intervals = self._add_and_thin(intervals, kernel)
			intervals = self._collapse_intervals(intervals)
			theta_tuples = []
			for i in intervals:
				theta_tuples.append(i[2].convert_to_tuple())
			return theta_tuples

		# finds critical points and tests optimal values
		def _get_new_intervals(self, interval, kernel):
			left, right, opt = interval

			# solve quadratic equation for critical points
			a = float(opt.q - kernel.q)
			b = 2 * float((kernel.q * kernel.m - opt.q * opt.m))
			c = float(opt.m**2 * opt.q - kernel.m**2 * kernel.q)
			endpoints = [left, right]
			if b**2 - 4 * a * c >= 0 and a != 0:
				pm = math.sqrt(b**2 - 4 * a * c)
				sol1 = (-1 * b + pm)/(2. * a)
				sol2 = (-1 * b - pm)/(2. * a)

				# add any valid solutions to list of endpoints
				if left < sol1 and sol1 < right:
					endpoints.append(sol1)
				if left < sol2 and sol2 < right:
					endpoints.append(sol2)
				endpoints.sort()
			return self._test_intervals(endpoints, kernel, opt)

		# test a point in each interval to find optimal kernel
		def _test_intervals(self, endpoints, kernel, opt):
			# go through each interval and use a test value
			new_intervals = []
			for k in range(len(endpoints) - 1):
				temp_left = endpoints[k]
				temp_right = endpoints[k + 1]
				test_val = (temp_left + temp_right)/2.
				new_opt = opt
				if kernel.evaluate(np.array([test_val])) > opt.evaluate(np.array([test_val])):
					new_opt = kernel
				new_intervals.append((temp_left, temp_right, new_opt))
			return new_intervals













		