"""
Check squeezing for different a12 values near Feschbach resonance.
Data is accumulated over several iterations which helps if many trajectories
are necessary and there is not enough GPU memory to process them at once.
"""

import numpy
from beclab import *
from beclab.meters import UncertaintyMeter, getXiSquared
import pickle


class AveragesCollector:

	def __init__(self, env, constants, grid):
		self._unc = UncertaintyMeter(env, constants, grid)
		self.times = []
		self.n1 = []
		self.n2 = []
		self.i = []

	def prepare(self, **kwds):
		self._unc.prepare(components=kwds['components'],
			ensembles=kwds['ensembles'], psi_type=kwds['psi_type'])

	def __call__(self, t, dt, psi):
		self.times.append(t)

		i, n = self._unc.getEnsembleSums(psi)

		self.i.append(i)
		self.n1.append(n[0])
		self.n2.append(n[1])

	def getData(self):
		return numpy.array(self.times), self.i, self.n1, self.n2


def testUncertainties(a12, gamma12, losses):

	t = 0.1
	callback_dt = 0.001
	N = 55000
	ensembles = 128

	parameters = dict(
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		a12=a12, a22=95.44,
		gamma111=0,
		gamma12=gamma12, gamma22=0)

	if not losses:
		parameters.update(dict(gamma111=0, gamma12=0, gamma22=0))

	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(), **parameters)
	grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

	gs = SplitStepGroundState(env, constants, grid, dt=1e-6)
	evolution = SplitStepEvolution(env, constants, grid, dt=1e-6)
	pulse = Pulse(env, constants, grid, f_rabi=350, f_detuning=-37)

	avc = AveragesCollector(env, constants, grid)

	psi = gs.create((N, 0))
	psi.toWigner(ensembles)

	pulse.apply(psi, math.pi / 2)

	evolution.run(psi, t, callbacks=[avc], callback_dt=callback_dt)
	env.release()

	times, i, n1, n2 = avc.getData()

	return times, i, n1, n2

def combinedTest(fname, N, a12, gamma12, losses):
	t = None
	ii = None
	nn1 = None
	nn2 = None

	for i in xrange(N):
		times, i, n1, n2 = testUncertainties(a12, gamma12, losses)

		if ii is None:
			ii = i
			nn1 = n1
			nn2 = n2
			t = times
		else:
			for j in xrange(len(ii)):
				ii[j] = numpy.concatenate([ii[j], i[j]])
				nn1[j] = numpy.concatenate([nn1[j], n1[j]])
				nn2[j] = numpy.concatenate([nn2[j], n2[j]])

		with open(fname, 'wb') as f:
			pickle.dump(
				dict(times=numpy.array(t),
					a12=a12, gamma12=gamma12, losses=losses, Is=numpy.array(ii),
					N1s=numpy.array(nn1), N2s=numpy.array(nn2)), f, protocol=2)


if __name__ == '__main__':

	iterations = 20

	params = [
		(80.0, 38.5e-19),
		(85.0, 19.3e-19),
		(90.0, 7.00e-19),
		(95.0, 0.853e-19)
	]

	import sys
	param_idx = int(sys.argv[1])
	losses = bool(int(sys.argv[2]))

	a12, gamma12 = params[param_idx]
	fname = 'feshbach_a12_' + str(a12) + ("" if losses else "_no_losses") + '.pickle'
	combinedTest(fname, iterations, a12, gamma12, losses)
