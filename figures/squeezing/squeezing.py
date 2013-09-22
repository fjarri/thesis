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


def testUncertainties(a12, gamma12):

	t = 0.1
	N = 55000
	ensembles = 2
	steps = 40000
	samples = 100

	env = envs.cuda(device_num=0)
	constants_kwds = dict(
		fx=97.0, fy=97.0 * 1.03, fz=11.69,
		a11=100.4, a12=a12, a22=95.44,
		gamma111=5.4e-42, gamma12=gamma12, gamma22=8.1e-20)

	constants = Constants(
		double=env.supportsDouble(),
		**constants_kwds)

	grid = UniformGrid.forN(env, constants, N, (64, 8, 8))

	gs = SplitStepGroundState(env, constants, grid, dt=1e-6)
	evolution = StrongRKEvolution(env, constants, grid)
	pulse = Pulse(env, constants, grid, f_rabi=350)

	avc = AveragesCollector(env, constants, grid)

	psi = gs.create((N, 0))
	psi.toWigner(ensembles)

	pulse.apply(psi, math.pi / 2)

	errors = evolution.run(psi, t, steps, callbacks=[avc], samples=samples)
	env.release()

	times, i, n1, n2 = avc.getData()

	return times, i, n1, n2

def combinedTest(N, a12, gamma12):
	t = None
	ii = None
	nn1 = None
	nn2 = None

	for i in xrange(N):
		print "*** Iteration", i

		times, i, n1, n2 = testUncertainties(a12, gamma12)

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

	return dict(
		times=numpy.array(t),
		Is=numpy.array(ii),
		N1s=numpy.array(nn1),
		N2s=numpy.array(nn2))


if __name__ == '__main__':

	iterations = 2

	params = [
		(80.0, 38.5e-19),
		(85.0, 19.3e-19),
		(90.0, 6.99e-19),
		(95.0, 0.853e-19)
	]

	for a12, gamma12 in params:
		xi = combinedTest(iterations, a12, gamma12)
		with open('feshbach_a12_' + str(a12) + '.pickle', 'wb') as f:
			pickle.dump(xi, f, protocol=2)
