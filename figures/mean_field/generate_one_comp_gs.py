import numpy
import json

from reikna import cluda

import beclab.constants as const
from beclab.bec import System, box_for_tf, HarmonicPotential
from beclab.grid import UniformGrid
from beclab.ground_state import tf_ground_state, it_ground_state


def get_gs(N):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    comp = const.rb87_1_minus1
    freqs = (97.6, 97.6, 11.96)
    shape = (32, 32, 128)
    dtype = numpy.complex128

    scattering = const.scattering_matrix([comp])
    potential = HarmonicPotential(freqs)
    system = System([comp], scattering, potential=potential)
    box = box_for_tf(system, 0, N)
    grid = UniformGrid(shape, box)

    tf_gs = tf_ground_state(thr, grid, dtype, system, [N])
    gs = it_ground_state(thr, grid, dtype, system, [N],
        E_diff=1e-7, E_conv=1e-9, sample_time=1e-5)

    n_x = (numpy.abs(gs.data.get()) ** 2)[:, 0].sum((1, 2)) * grid.dxs[0] * grid.dxs[1]
    tf_n_x = (numpy.abs(tf_gs.data.get()) ** 2)[:, 0].sum((1, 2)) * grid.dxs[0] * grid.dxs[1]
    xs = grid.xs[2]

    return xs.tolist(), n_x.tolist(), tf_n_x.tolist()


if __name__ == '__main__':

    result = {}

    for N in (1000, 100000):
        xs, n_x, tf_n_x = get_gs(N)
        result[N] = dict(xs=xs, n_x=n_x, tf_n_x=tf_n_x)

    with open('one_comp_gs.json', 'w') as f:
        json.dump(result, f, indent=4)
