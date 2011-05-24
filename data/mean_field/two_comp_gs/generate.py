import numpy
import time
import math

from beclab import *
from beclab.state import Projection


def getGS(N):

	# preparation
	env = envs.cpu()
	constants = Constants(Model(N=N, detuning=-41, dt_steady=1e-5,
		a11=100.4, a12=98.13, a22=95.68),
		double=True)

	gs = GPEGroundState(env, constants)
	p = Projection(env, constants)

	cloud = gs.createCloud(two_component=True, precision=1e-8)

	z = (-constants.zmax + numpy.arange(constants.nvz) * constants.dz) * 1e6 # in microns
	a_z = p.getZ(cloud.a) * constants.dx * constants.dy / 1e6
	b_z = p.getZ(cloud.b) * constants.dx * constants.dy / 1e6

	a_data = XYData("|1>",
		z, a_z, ymin=0, xname="z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)")
	b_data = XYData("|2>",
		z, b_z, ymin=0, xname="z ($\\mu$m)", yname="Axial density ($\\mu$m$^{-1}$)")

	env.release()
	return a_data, b_data

a, b = getGS(80000)
a.save("gs_a_80k.json")
b.save("gs_b_80k.json")
