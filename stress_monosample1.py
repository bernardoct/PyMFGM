import numpy as np
from math import ceil, floor
from scipy.stats import pearsonr
from scipy.linalg import cholesky
import os
import argparse
from multiprocessing import Pool, Manager
from functools import partial
import time
# from guppy import hpy
# import gc
# from mpi4py import MPI

def corr(a, b):
	d = len(a[0])
	c = np.zeros([d, d])

	for i in range(d):
		for j in range(d):
			c[i,j] = pearsonr(a[:, i].T, b[:, j].T)[0]

	return c

# @profile
def generate_one_synthetic(Random_Matrix, num_years, Q_matrix_int):
	Qs_uncorr = np.zeros([num_years, 52])
	weekly_mean = np.zeros(52)
	weekly_stdev = np.zeros(52)
	nQ_historical = Q_matrix_int.shape[0]
	nQ = nQ_historical * 100

	for yr in range(num_years):

		Q_matrix = np.tile(Q_matrix_int, (100, 1))

		logQ = np.log(Q_matrix)
		logQint = np.log(Q_matrix_int)
		Z = np.zeros([int(nQ), 52])

		for i in range(52):
			weekly_mean[i] = np.mean(logQint[:, i])
			weekly_stdev[i] = np.std(logQint[:, i], ddof=1)
			Z[:, i] = (logQ[:, i] - weekly_mean[i]) / weekly_stdev[i]
		
		for i in range(52):
			# print Random_Matrix[yr, i].shape, Z.shape, Qs_uncorr.shape
			Qs_uncorr[yr, i] = Z[int(round(Random_Matrix[yr, i])), i]

	Z_vector = Z.ravel()
	Z_shifted = np.reshape(Z_vector[26:nQ * 52 - 26], (52, -1), 'F').T
#	 The correlation matrices should use the historical Z's
#	 (the "appended years" do not preserve correlation)
	Z_hist = Z[0:nQ_historical, :]
	Z_hist_shifted = Z_shifted[0:nQ_historical - 1, :]
	U = cholesky(corr(Z_hist, Z_hist), lower=False)
	U_shifted = cholesky(corr(Z_hist_shifted, Z_hist_shifted), lower=False)

	Qs_uncorr_vector = Qs_uncorr.ravel()
	Qs_uncorr_shifted = np.reshape(Qs_uncorr_vector[26:num_years * 52 - 26], (52, -1), 'F').T

	Qs_corr = np.dot(Qs_uncorr, U)
	Qs_corr_shifted = np.dot(Qs_uncorr_shifted, U_shifted)

	Qs_log = np.zeros([num_years - 1, num_years - 1])
	Qs_log[:, 0:26] = Qs_corr_shifted[:,26:52]
	Qs_log[:, 26:52] = Qs_corr[1:num_years, 26:52]

	return Qs_log

	# Qsk = np.zeros([num_years - 1, 52])
	# for year in range(num_years-1):
	# 	for i in range(52):
	# 		Qsk[year, i] = np.exp(Qs_log[year, i] * weekly_stdev[i] + weekly_mean[i])
	# 		# print Qsk[year, i], Qs_log[year, i] * weekly_stdev[i], weekly_mean[i] 
	# return Qsk

# @profile
def stress_dynamic(E_historical, Q_historical, num_years):

	Qs = []
	Es = []

	if isinstance(Q_historical, list):
		npoints = len(Q_historical)
	else:
		raise Exception('Q_historical must be a list containing one or more 2-D numpy matrices.')

	if len(Q_historical) != len(E_historical):
		raise Exception('There must be one evaporation series for each inflow series')
	
	for i in range(1, npoints):
		if Q_historical[i].shape != Q_historical[0].shape:
			raise Exception('All matrices in Q_historical must be the same size.')
		if E_historical[i].shape != E_historical[0].shape:
			raise Exception('All matrices in E_historical must be the same size.')
		if Q_historical[i].shape != E_historical[0].shape:
			raise Exception('All matrices in Q_historical and E_historical must be the same size.')

	num_years += 1 # adjusts for the new corr technique
	Random_Matrix = np.zeros([num_years, 52])
	nQ_historical = Q_historical[0].shape[0]
	nQ = nQ_historical * 100

	# Random matrix for samping weeks.
	for yr in range(num_years):
		Random_Matrix[yr, :] = np.random.randint(int(nQ), size=52)

	for k in range(npoints):
		Qs.append(generate_one_synthetic(Random_Matrix, num_years, Q_historical[k]))
		Es.append(generate_one_synthetic(Random_Matrix, num_years, E_historical[k]))
	
	return Qs, Es

# def realization(m, r):
def realization(Qh, num_years, r):
	print 'running realization ' + str(r)
	time1 = time.time()
	Qs = stress_dynamic(Qh, num_years) # or call stress(Qh, num_years, p, n)
	# Qs = stress_dynamic(m[0], m[1]) # or call stress(Qh, num_years, p, n)

	time2 = time.time()
	print 'stress_dynamic took %0.3f s' % ((time2-time1))

	return Qs

#def gen_sample(args):
# @profile
def gen_sample(sample_no, num_realizations):

	inflow_dir = 'inflow-data_updated'
	inflow_files = ['claytonGageInflow', 'crabtreeCreekInflow', 'updatedFallsLakeInflow', 'updatedJordanLakeInflow', 'updatedLakeWBInflow', 'updatedLillingtonInflow', 'updatedLittleRiverInflow', 'updatedLittleRiverRaleighInflow', 'updatedMichieInflow', 'updatedOWASAInflow']

	num_years = 70

	Qh = []
	output = []
	for k in range(len(inflow_files)):
		hyd_data = np.loadtxt(inflow_dir + '/' + inflow_files[k] + '.csv', delimiter=',')
		Qh.append(hyd_data)
		output.append(np.zeros([num_realizations, num_years * 52]))


	if not os.path.exists('inflow-synthetic/'):
	    os.makedirs('inflow-synthetic/')

	# m = Manager().list()
	# m.append(Qh)
	# m.append(num_years)
	# partial_realization = partial(realization, m)
	# output_parallel = Pool().map(partial_realization, range(num_realizations))
	partial_realization = partial(realization, Qh, num_years)
	output_parallel = Pool().map(partial_realization, range(num_realizations))

	for r in range(num_realizations):
		for k in range(len(inflow_files)):
			output[k][r, :] = output_parallel[r][k].ravel()

	# for r in range(num_realizations):
	# 	print 'running realization ' + str(r)
	# 	time1 = time.time()
	# 	Qs = stress_dynamic(Qh, num_years) # or call stress(Qh, num_years, p, n)

	# 	for k in range(len(inflow_files)):
	# 		output[k][r, :] = Qs[k].ravel()

	# 	time2 = time.time()
	# 	print 'stress_dynamic took %0.3f s' % ((time2-time1))

	for k in range(len(inflow_files)):
		np.savetxt('inflow-synthetic/' + inflow_files[k] + 'SYN' + str(int(sample_no)) + '.csv', output[k], delimiter=',')

if __name__ == '__main__':
# comm = MPI.COMM_WORLD

# print comm.rank
# gen_sample(comm.rank, 1000)
	gen_sample(0, 1000)
