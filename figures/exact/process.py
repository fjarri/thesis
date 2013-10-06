from runxmds import get, getn, deln
import os.path
import itertools
import os.path
import pickle

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Load data from Qiongyi's txt files
def load_txt_data(fname):
	f = open(os.path.join('Qiongyi_data', fname))
	lines = f.readlines()
	f.close()

	parsed = [
		[float(x) if x != '--' else float('NaN') for x in line.split()]
		for line in lines]
	for l in parsed:
		for x in l:
			if type(x) != type(1.0):
				print x

	return numpy.array(parsed).T



# Qiongyi's plots
tau_c, s_c, s_pi2_c, tau_nc, s_nc, s_pi2_nc = load_txt_data('squeezingviaBEC.txt')

with open('single_well_squeezing_exact.pickle', 'w') as f:
	pickle.dump(dict(
		tau_c=tau_c,
		s_pi2_c=s_pi2_c,
		tau_nc=tau_nc,
		s_pi2_nc=s_pi2_nc,
		), f, protocol=2)


tmax = {
	(20, True): 36,
	(200, True): 120,
	(2000, True): 360,
	(20, False): 6,
	(200, False): 20,
	(2000, False): 60}

subsets = 20

for Na, trajectories in [
		(200, '100'),
		(200, '1k'),
		(200, '10k'),
		(20, '10k'),
		(2000, '10k')
		]:

	#deln('epr_wigner_' + trajectories, subsets,
	#	Na=Na, Nb=Na, B=9.116, kappa1_t=0, kappa2_t=0, losses=0, tau_max=tmax[(Na, True)])
	#deln('epr_wigner_' + trajectories, subsets,
	#	Na=Na, Nb=Na, B=0, a11=100.4, a22=100.4, kappa1_t=0, kappa2_t=0, losses=0,
	#		tau_max=tmax[(Na, False)])

	results_cc = getn('epr_wigner_' + trajectories, subsets,
		Na=Na, Nb=Na, B=9.116, kappa1_t=0, kappa2_t=0, losses=0, tau_max=tmax[(Na, True)])
	results_nocc = getn('epr_wigner_' + trajectories, subsets,
		Na=Na, Nb=Na, B=0, a11=100.4, a22=100.4, kappa1_t=0, kappa2_t=0, losses=0,
			tau_max=tmax[(Na, False)])

	S_pi2_cc_arr = numpy.array([x.sw.S_pi2 for x in results_cc])
	S_pi2_nocc_arr = numpy.array([x.sw.S_pi2 for x in results_nocc])

	with open('single_well_squeezing_wigner_' + trajectories + '_Na' + str(Na) + '.pickle', 'w') as f:
		pickle.dump(dict(
			tau_c=results_cc[0].tau,
			s_pi2_c=S_pi2_cc_arr.mean(0),
			s_pi2_c_err=S_pi2_cc_arr.std(0) / numpy.sqrt(subsets),
			tau_nc=results_nocc[0].tau,
			s_pi2_nc=S_pi2_nocc_arr.mean(0),
			s_pi2_nc_err=S_pi2_nocc_arr.std(0) / numpy.sqrt(subsets),
			), f, protocol=2)

