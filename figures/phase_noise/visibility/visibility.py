import numpy
import time
import pickle

from beclab import *
from beclab.evolution_strong import StrongRKEvolution
import gc
import sys


def calculateRamsey(wigner=False, echo=False, t=1.0,
		steps=20000, samples=100, N=55000, ensembles=1, shape=(64, 8, 8)):

	env = envs.cuda(device_num=0)
	constants_kwds = dict(
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		a11=100.4, a12=98.0, a22=95.44,
		gamma111=5.4e-42, gamma12=1.51e-20, gamma22=8.1e-20)

	constants = Constants(
		double=env.supportsDouble(),
		**constants_kwds)

	grid = UniformGrid.forN(env, constants, N, shape)

	gs = SplitStepGroundState(env, constants, grid, dt=1e-6)
	evolution = StrongRKEvolution(env, constants, grid)
	pulse = Pulse(env, constants, grid, f_rabi=350)

	n = WavefunctionCollector(env, constants, grid)

	collectors = [n]

	psi = gs.create((N, 0), precision=1e-6)

	if wigner:
		psi.toWigner(ensembles)

	pulse.apply(psi, numpy.pi / 2)

	if t > 0:
		t1 = time.time()
		if echo:
			errors = evolution.run(psi, t / 2, steps / 2)
			pulse.apply(psi, numpy.pi)
			errors = evolution.run(psi, t / 2, steps / 2)
		else:
			errors = evolution.run(psi, t, steps, callbacks=collectors, samples=samples)
		env.synchronize()
		t2 = time.time()
		print "Time spent: " + str(t2 - t1) + " s"
	else:
		errors = None

	if echo:
		psis = psi.data.get()
		times = t
	else:
		times, psis = n.getData()

	psi_type = psi.type

	del psi
	del evolution
	del gs
	del pulse
	del constants
	del grid
	env.release()
	del env
	gc.collect()

	return dict(times=times, psis=psis, errors=errors, psi_type=psi_type,
		constants_kwds=constants_kwds, N=N, steps=steps, shape=shape)


def calculateEcho(**kwds):
	kwds['echo'] = True
	total_steps = kwds['steps']
	total_samples = kwds['samples']
	total_t = float(kwds['t'])
	assert total_steps % total_samples == 0

	ress = None
	for j in xrange(total_samples + 1):
		steps = total_steps / total_samples * j
		t = total_t / total_samples * j
		print "--- Running Ramsey for t =", t, " steps =", steps

		kwds['t'] = t
		kwds['steps'] = steps
		res = calculateRamsey(**kwds)
		if ress is None:
			ress = res
			ress['times'] = [t]
			ress['psis'] = [res['psis']]
		else:
			ress['times'].append(t)
			ress['psis'].append(res['psis'])

	ress['errors'] = res['errors']
	return ress


def run(func, fname, ens_step, **kwds):

	total_ensembles = kwds['ensembles']
	kwds['ensembles'] = ens_step

	for j in xrange(0, total_ensembles, ens_step):
		print "*** Ensembles:", j, "to", j + ens_step - 1
		res = func(**kwds)

		if j > 0:
			with open(fname, 'rb') as f:
				ress = pickle.load(f)

			res['errors']['error_strong_max'] = max(
				ress['errors']['error_strong_max'],
				res['errors']['error_strong_max'])
			res['errors']['error_weak_max'] = max(
				ress['errors']['error_weak_max'],
				res['errors']['error_weak_max'])

			psis = ress['psis']
			psis_part = res['psis']

			psis_new = []
			for psi, psi_part in zip(psis, psis_part):
				psi = psi.transpose(*((1, 0) + tuple(range(2, len(psi.shape)))))
				psi_part = psi_part.transpose(*((1, 0) + tuple(range(2, len(psi_part.shape)))))
				psi_new = numpy.vstack([psi, psi_part])
				psi_new = psi_new.transpose(*((1, 0) + tuple(range(2, len(psi_new.shape)))))
				psis_new.append(psi_new)

			res['psis'] = psis_new

		with open(fname, 'wb') as f:
			pickle.dump(res, f, protocol=2)


if __name__ == '__main__':

	# Short time
	"""
	#if sys.argv[1] == 'ramsey-gpe':
	run(calculateRamsey, 'ramsey_gpe.pickle', 1,
		t=1.3, steps=40000, samples=100, N=55000, wigner=False, ensembles=1, shape=(64,8,8))
	#elif sys.argv[1] == 'echo-gpe':
	run(calculateEcho, 'echo_gpe.pickle', 1,
		t=1.8, steps=40000, samples=50, N=55000, wigner=False, ensembles=1, shape=(64,8,8))
	#elif sys.argv[1] == 'ramsey-wigner':
	run(calculateRamsey, 'ramsey_wigner.pickle', 8,
		t=1.3, steps=80000, samples=100, N=55000, wigner=True, ensembles=64, shape=(64,8,8))
	#elif sys.argv[1] == 'echo-wigner':
	run(calculateEcho, 'echo_wigner.pickle', 8,
		t=1.8, steps=80000, samples=50, N=55000, wigner=True, ensembles=64, shape=(64,8,8))
	"""

	# Long time
	#if sys.argv[1] == 'ramsey-gpe':
	run(calculateRamsey, 'ramsey_long_gpe.pickle', 1,
		t=5, steps=160000, samples=100, N=55000, wigner=False, ensembles=1, shape=(64,8,8))
	#elif sys.argv[1] == 'echo-gpe':
	run(calculateEcho, 'echo_long_gpe.pickle', 1,
		t=5, steps=160000, samples=40, N=55000, wigner=False, ensembles=1, shape=(64,8,8))
	#elif sys.argv[1] == 'ramsey-wigner':
	run(calculateRamsey, 'ramsey_long_wigner.pickle', 16,
		t=6, steps=320000, samples=100, N=55000, wigner=True, ensembles=32, shape=(64,8,8))
	#elif sys.argv[1] == 'echo-wigner':
	run(calculateEcho, 'echo_long_wigner.pickle', 16,
		t=5, steps=320000, samples=40, N=55000, wigner=True, ensembles=32, shape=(64,8,8))
