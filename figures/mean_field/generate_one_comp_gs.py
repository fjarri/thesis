import numpy
import pickle

from beclab import *
from beclab.meters import ProjectionMeter
from beclab.ground_state import TFGroundState


def getGS(N):

	# preparation
	env = envs.cuda()
	constants = Constants(double=env.supportsDouble(), a11=100.4, a12=97.99, a22=95.44)
	grid = UniformGrid.forN(env, constants, N, (128, 32, 32))

	tf_gs = TFGroundState(env, constants, grid)
	gs = SplitStepGroundState(env, constants, grid, dt=1e-5)

	psi = gs.create((N,), precision=1e-6)
	psi_tf = tf_gs.create((N,))

	prj = ProjectionMeter.forPsi(psi)

	# Projection on Z axis
	n_z = prj.getZ(psi)
	tf_n_z = prj.getZ(psi_tf)
	zs = grid.z

	env.release()

	return zs, n_z, tf_n_z


if __name__ == '__main__':

	result = {}

	for N in (1000, 100000):
		zs, n_z, tf_n_z = getGS(N)
		result[N] = dict(zs=zs, n_z=n_z, tf_n_z=tf_n_z)

	with open('one_comp_gs.pickle', 'wb') as f:
		pickle.dump(result, f, protocol=2)
