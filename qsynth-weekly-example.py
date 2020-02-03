import os
from pymfgm import * 


sample_data_dir = './inflow-data_updated'
sample_name = 'inflow-weekly-seq'
num_realizations = [1, 10,] #100, 1000, 10000]

# load historical hydro data
inflow_files = [os.path.join(sample_data_dir,f) \
 for f in os.listdir(sample_data_dir) if f.endswith('.csv')]
Qhistorical = load_hydro(inflow_files)

# make sure an synthetic output exists
outdir = os.path.join('./my_synthetic_generations',sample_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)


times = [] # time it
for idx,sample_size in enumerate(num_realizations):
	start = time.perf_counter() # start clock

	#-----------magic happens----------#
	# generate synthetic data
	Qsynthetic = gen_sample(
		Qh=Qhistorical, 
		num_sites=len(inflow_files), 
		num_realizations=sample_size,
		num_years=51,
		freq=52,
		# n_proc=4
	)
	# write out synthetic data
	for idx,k in enumerate(inflow_files):
		location_name = os.path.basename(k).split('.')[0]
		np.savetxt(
			os.path.join(outdir,
				location_name+'SYN'+str(int(sample_size))+'.csv') ,
			Qsynthetic[idx], 
			delimiter=','
		)
	#------------in here----------------#

	end = time.perf_counter() # end clock
	times.append([round(end - start,1), sample_size]) # log time

np.savetxt('times' + '.csv', times, delimiter=',')

