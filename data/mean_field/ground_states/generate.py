import numpy
import time
import math

from beclab import *
from beclab.state import Projection
from beclab.ground_state import TFGroundState


def getGS(N):

	# preparation
	env = envs.cpu()
	constants = Constants(Model(N=N, detuning=-41, dt_steady=1e-5,
		a11=100.4, a12=97.99, a22=95.57),
		double=True)

	gs = GPEGroundState(env, constants)
	tf_gs = TFGroundState(env, constants)

	p = Projection(env, constants)

	cloud = gs.createCloud()
	tf_a = tf_gs.create(comp=cloud.a.comp, N=N)

	z = (-constants.zmax + numpy.arange(constants.nvz) * constants.dz) * 1e6 # in microns
	a_z = p.getZ(cloud.a) * constants.dx * constants.dy / 1e6
	tf_a_z = p.getZ(tf_a) * constants.dx * constants.dy / 1e6

	a_data = XYData("Accurate",
		z, a_z, ymin=0, xname="z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)")
	tf_a_data = XYData("Thomas-Fermi approximation",
		z, tf_a_z, ymin=0, xname="z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)")

	env.release()
	return a_data, tf_a_data

for N in (1000, 100000):
	a, tf_a = getGS(N)
	a.save("gs_" + str(int(N / 1000)) + "k.json")
	tf_a.save("tf_gs_" + str(int(N / 1000)) + "k.json")
