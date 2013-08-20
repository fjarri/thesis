import numpy
import pickle

from beclab import *
from beclab.meters import ProjectionMeter


def getGS(N, a12):

	# preparation
	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(), a11=100.4, a12=a12, a22=95.44)
	grid = UniformGrid.forN(env, constants, N, (128, 32, 32))

	tf_gs = TFGroundState(env, constants, grid)
	gs = SplitStepGroundState(env, constants, grid, dt=1e-5)

	psi = gs.create((N/2,N/2), precision=1e-8)

	prj = ProjectionMeter.forPsi(psi)

	# Projection on Z axis
	n_z = prj.getZ(psi)
	zs = grid.z

	env.release()

	return zs, n_z


if __name__ == '__main__':

	result = {}

	for a12 in (97.0, 99.0):
		zs, n_z = getGS(80000, a12)
		result[a12] = dict(zs=zs, n_z=n_z)

	with open('two_comp_gs.pickle', 'wb') as f:
		pickle.dump(result, f, protocol=2)
