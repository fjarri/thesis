import numpy
import json

from reikna import cluda

import beclab.constants as const
from beclab.bec import System, box_for_tf, HarmonicPotential
from beclab.grid import UniformGrid
from beclab.ground_state import tf_ground_state, it_ground_state


def get_gs(N, a12):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    comps = [const.rb87_1_minus1, const.rb87_2_1]
    freqs = (97.6, 97.6, 11.96)
    shape = (32, 32, 128)
    dtype = numpy.complex128

    scattering = const.scattering_3d(numpy.array([[100.4, a12], [a12, 95.44]]), comps[0].m)
    potential = HarmonicPotential(freqs)
    system = System(comps, scattering, potential=potential)
    box = box_for_tf(system, 0, N)
    grid = UniformGrid(shape, box)

    gs = it_ground_state(thr, grid, dtype, system, [N/2, N/2],
        E_diff=1e-7, E_conv=1e-9, sample_time=1e-4)

    n_x = (numpy.abs(gs.data.get()) ** 2)[:, 0].sum((1, 2)) * grid.dxs[0] * grid.dxs[1]
    xs = grid.xs[2]

    return xs.tolist(), n_x.tolist()


if __name__ == '__main__':

	result = {}

	for a12 in (97.0, 99.0):
		xs, n_x = get_gs(80000, a12)
		result[a12] = dict(xs=xs, n_x=n_x)

	with open('two_comp_gs.json', 'w') as f:
		json.dump(result, f, indent=4)
