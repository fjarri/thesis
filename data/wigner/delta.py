"""
Builds plots for "restricted" delta-functions in 1D
"""

from functools import reduce as freduce

import numpy
from numpy.polynomial import Hermite as H
from scipy.interpolate import griddata

from beclab import *
from beclab.helpers.misc import tile2D


def factorial(n):
	res = 1
	for i in xrange(2, n + 1):
		res *= i
	return res

def eigenf_uniform(constants, grid, n):
	k = grid.kz
	V = grid.V
	def func(z):
		return numpy.exp(1j * k[n] * z) / numpy.sqrt(V)

	return func

def eigenf_harmonic(constants, grid, n):
	l = grid.lz
	def func(z):
		return H([0] * n + [1])(z / l) / (numpy.pi ** 0.25) / \
			numpy.sqrt(float(long(2 ** n) * factorial(n))) * \
			numpy.exp(-((z / l) ** 2) / 2) / numpy.sqrt(l)

	return func

def getDeltaPlot(basis, e_cut):

	uniform_modes = 256
	harmonic_modes = 40

	env = envs.cpu()
	constants = Constants(double=True, use_effective_area=True, e_cut=e_cut)

	if basis == 'uniform':
		# setting box size based on HarmonicGrid box,
		# in order to make sizes of plots equal
		grid_harmonic = HarmonicGrid(env, constants, (harmonic_modes,))
		grid = UniformGrid(env, constants, (uniform_modes,),
			(grid_harmonic.z[-1] - grid_harmonic.z[0],))

		modenums = (numpy.fft.fftfreq(grid.shape[0], 1) * grid.shape[0]).astype(numpy.int32)
		eigenf = eigenf_uniform
	else:
		grid = HarmonicGrid(env, constants, (harmonic_modes,))
		modenums = numpy.arange(grid.shape[0])
		eigenf = eigenf_harmonic

	mask = grid.projector_mask
	z = grid.z

	delta = numpy.zeros((grid.shape[0], grid.shape[0])).astype(numpy.complex128)
	for n in xrange(mask.size):
		if mask[n] == 0:
			continue

		mode = modenums[n]
		ef = eigenf(constants, grid, mode)(z)

		z1, z2 = tile2D(ef, ef)
		delta += z1 * z2.conj()

	delta = numpy.abs(delta)

	if basis == 'harmonic':
	# Interpolate data to uniform grid
		z0x, z0y = tile2D(grid.z, grid.z)
		z1 = numpy.linspace(grid.z[0], grid.z[-1], uniform_modes)
		z2 = numpy.linspace(grid.z[0], grid.z[-1], uniform_modes)
		delta = griddata((z0x.ravel(), z0y.ravel()), delta.ravel(), (z1[None,:], z2[:,None]), method='cubic')

	return HeightmapPlot(HeightmapData('test', delta,
		xmin=grid.z[0] * 1e6, xmax=grid.z[-1] * 1e6,
		ymin=grid.z[0] * 1e6, ymax=grid.z[-1] * 1e6,
		xname="$z$ ($\\mu$m)", yname="$z^\\prime$ ($\\mu$m)",
		zmin=0), colorbar=False)


if __name__ == '__main__':
	getDeltaPlot('harmonic', 1000).save('delta_harmonic_1000.eps')
	getDeltaPlot('uniform', 1000).save('delta_uniform_1000.eps')
	getDeltaPlot('harmonic', None).save('delta_harmonic_all.eps')
	getDeltaPlot('uniform', None).save('delta_uniform_all.eps')
