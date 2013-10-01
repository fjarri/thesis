from runxmds import get, getn
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




N = 20

results_cc = getn('epr_wigner_100', N, Na=200, Nb=200, B=9.116, kappa1_t=0, kappa2_t=0, losses=0, tau_max=120)
results_nocc = getn('epr_wigner_100', N, Na=200, Nb=200, B=0, a11=100.4, a22=100.4, kappa1_t=0, kappa2_t=0, losses=0, tau_max=25)

S_pi2_cc_arr = numpy.array([x.sw.S_pi2 for x in results_cc])
S_pi2_nocc_arr = numpy.array([x.sw.S_pi2 for x in results_nocc])


with open('single_well_squeezing_wigner_100.pickle', 'w') as f:
	pickle.dump(dict(
		tau_c=results_cc[0].tau,
		s_pi2_c=S_pi2_cc_arr.mean(0),
		s_pi2_c_err=S_pi2_cc_arr.std(0) / numpy.sqrt(N),
		tau_nc=results_nocc[0].tau,
		s_pi2_nc=S_pi2_nocc_arr.mean(0),
		s_pi2_nc_err=S_pi2_nocc_arr.std(0) / numpy.sqrt(N),
		), f, protocol=2)


results_cc = getn('epr_wigner_1k', N, Na=200, Nb=200, B=9.116, kappa1_t=0, kappa2_t=0, losses=0, tau_max=120)
results_nocc = getn('epr_wigner_1k', N, Na=200, Nb=200, B=0, a11=100.4, a22=100.4, kappa1_t=0, kappa2_t=0, losses=0, tau_max=25)

S_pi2_cc_arr = numpy.array([x.sw.S_pi2 for x in results_cc])
S_pi2_nocc_arr = numpy.array([x.sw.S_pi2 for x in results_nocc])


with open('single_well_squeezing_wigner_1k.pickle', 'w') as f:
	pickle.dump(dict(
		tau_c=results_cc[0].tau,
		s_pi2_c=S_pi2_cc_arr.mean(0),
		s_pi2_c_err=S_pi2_cc_arr.std(0) / numpy.sqrt(N),
		tau_nc=results_nocc[0].tau,
		s_pi2_nc=S_pi2_nocc_arr.mean(0),
		s_pi2_nc_err=S_pi2_nocc_arr.std(0) / numpy.sqrt(N),
		), f, protocol=2)


results_cc = getn('epr_wigner_10k', N, Na=200, Nb=200, B=9.116, kappa1_t=0, kappa2_t=0, losses=0, tau_max=120)
results_nocc = getn('epr_wigner_10k', N, Na=200, Nb=200, B=0, a11=100.4, a22=100.4, kappa1_t=0, kappa2_t=0, losses=0, tau_max=25)

S_pi2_cc_arr = numpy.array([x.sw.S_pi2 for x in results_cc])
S_pi2_nocc_arr = numpy.array([x.sw.S_pi2 for x in results_nocc])


with open('single_well_squeezing_wigner_10k.pickle', 'w') as f:
	pickle.dump(dict(
		tau_c=results_cc[0].tau,
		s_pi2_c=S_pi2_cc_arr.mean(0),
		s_pi2_c_err=S_pi2_cc_arr.std(0) / numpy.sqrt(N),
		tau_nc=results_nocc[0].tau,
		s_pi2_nc=S_pi2_nocc_arr.mean(0),
		s_pi2_nc_err=S_pi2_nocc_arr.std(0) / numpy.sqrt(N),
		), f, protocol=2)

