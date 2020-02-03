from __future__ import print_function
import numpy as np
from math import ceil, floor
from scipy.stats import pearsonr
from scipy.linalg import cholesky
import argparse
from multiprocessing import Pool, Manager
from functools import partial
import time

def corr(a, b):
	d = len(a[0])
	c = np.zeros([d, d])

	for i in range(d):
		for j in range(d):
			c[i,j] = pearsonr(a[:, i].T, b[:, j].T)[0]

	return c

# @profile
def generate_one_synthetic(Random_Matrix, num_years, Q_matrix_int, freq):
	Qs_uncorr = np.zeros([num_years, freq])
	weekly_mean = np.zeros(freq)
	weekly_stdev = np.zeros(freq)
	nQ_historical = Q_matrix_int.shape[0]
	nQ = nQ_historical * 100
	halffreq = int(freq/2)
	for yr in range(num_years):

		Q_matrix = np.tile(Q_matrix_int, (100, 1))

		logQ = np.log(Q_matrix)
		logQint = np.log(Q_matrix_int)
		Z = np.zeros([int(nQ), freq])

		for i in range(freq):
			weekly_mean[i] = np.mean(logQint[:, i])
			weekly_stdev[i] = np.std(logQint[:, i], ddof=1)
			Z[:, i] = (logQ[:, i] - weekly_mean[i]) / weekly_stdev[i]
		
		for i in range(freq):
			Qs_uncorr[yr, i] = Z[int(round(Random_Matrix[yr, i])), i]

	Z_vector = Z.ravel()
	Z_shifted = np.reshape(
		Z_vector[halffreq:nQ * freq - halffreq], (freq, -1), 'F').T

	# The correlation matrices should use the historical Z's
	# (the "appended years" do not preserve correlation)
	Z_hist = Z[0:nQ_historical, :]
	Z_hist_shifted = Z_shifted[0:nQ_historical - 1, :]

	U = cholesky(corr(Z_hist, Z_hist), lower=False)
	U_shifted = cholesky(corr(Z_hist_shifted, Z_hist_shifted), lower=False)

	Qs_uncorr_vector = Qs_uncorr.ravel()
	Qs_uncorr_shifted = np.reshape(
		Qs_uncorr_vector[halffreq:num_years * freq - halffreq], (freq, -1), 'F').T

	Qs_corr = np.dot(Qs_uncorr, U)
	Qs_corr_shifted = np.dot(Qs_uncorr_shifted, U_shifted)

	Qs_log = np.zeros([num_years-1, freq])
	Qs_log[:, 0:halffreq] = Qs_corr_shifted[:,halffreq:freq]
	Qs_log[:, halffreq:freq] = Qs_corr[1:num_years, halffreq:freq]

	Qsk = np.zeros([num_years - 1, freq])
	for year in range(num_years-1):
		for i in range(freq):
			Qsk[year, i] = np.exp(Qs_log[year, i] * weekly_stdev[i] + weekly_mean[i])
	return Qsk

# @profile
def stress_dynamic(Q_historical, num_years, freq):

	Qs = []

	if isinstance(Q_historical, list):
		npoints = len(Q_historical)
	else:
		raise Exception('Q_historical must be a list containing one or more 2-D numpy matrices.')
	
	for i in range(1, npoints):
		if Q_historical[i].shape != Q_historical[0].shape:
			raise Exception('All matrices in Q_historical must be the same size.');

	num_years += 1 # adjusts for the new corr technique
	Random_Matrix = np.zeros([num_years, freq])
	nQ_historical = Q_historical[0].shape[0]
	nQ = nQ_historical * 100

	# Random matrix for samping weeks.
	for yr in range(num_years):
		Random_Matrix[yr, :] = np.random.randint(int(nQ), size=freq)

	for k in range(npoints):
		Qs.append(
			generate_one_synthetic(Random_Matrix, num_years, Q_historical[k], freq)
		)
	
	return Qs

# def realization(m, r):
def realization(Qh, num_years, freq, r):
	np.random.seed(r) # !IMPORTANT(otherwise all subprocess will have same seed)!
	Qs = stress_dynamic(Qh, num_years, freq) # or call stress(Qh, num_years, p, n)
	return Qs

#def gen_sample(args):
# @profile
def gen_sample(Qh, num_sites, num_realizations, num_years, freq, n_proc=1):

	output = []
	for k in range(num_sites):
		output.append(np.zeros([num_realizations, num_years * freq]))

	if n_proc > 1:

		pool = Pool(processes=n_proc)
		partial_realization = partial(realization, Qh, num_years, freq)
		output_parallel = pool.map(partial_realization, range(num_realizations))

		for r in range(num_realizations):
			for k in range(num_sites):
				output[k][r, :] = output_parallel[r][k].ravel()

	else:

		for r in range(num_realizations):
			print('running realization ' + str(r))
			time1 = time.time()

			np.random.seed(r)
			Qs = stress_dynamic(Qh, num_years, freq) # or call stress(Qh, num_years, p, n)

			for k in range(num_sites):
				output[k][r, :] = Qs[k].ravel()

			time2 = time.time()
			print('stress_dynamic took %0.3f s' % ((time2-time1)))

	return output

def load_hydro(inflow_files):

	Qh = []

	for k in inflow_files:
		hyd_data = np.loadtxt(k, delimiter=',')
		Qh.append(hyd_data)

	return Qh
