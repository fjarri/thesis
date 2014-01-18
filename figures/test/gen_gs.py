import numpy
import time
import pickle

from reikna import cluda

from beclab import *


def generate_gs(N=55000, shape=(8,8,64), box_modifiers=None):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    freqs = (97.0, 97.0 * 1.03, 11.69)
    dtype = numpy.complex128

    comps = [const.rb87_1_minus1, const.rb87_2_1]
    scattering = const.scattering_3d(numpy.array([[100.4]]), comps[0].m)

    potential = HarmonicPotential(freqs)
    system = System([comps[0]], scattering, potential=potential)

    reference_grid = UniformGrid((8,8,64), box_for_tf(system, 0, N))
    if box_modifiers is None:
        box_modifiers = (1, 1, 1)

    box = reference_grid.box
    new_box = tuple(d * m for d, m in zip(box, box_modifiers))
    grid = UniformGrid(shape, new_box)
    print(grid.shape, grid.box)

    it_gs_gen = ImaginaryTimeGroundState(thr, dtype, grid, system)
    gs = it_gs_gen([N], E_diff=1e-7, E_conv=1e-9, sample_time=1e-5)

    psi = gs.data.get()
    psi = numpy.concatenate([psi, numpy.zeros_like(psi)], axis=1)

    fname = 'ground_states/ground_state_' + '-'.join([str(s) for s in shape]) + \
        '_' + '-'.join([str(b) for b in box_modifiers]) + '.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(dict(
            data=psi,
            shape=grid.shape,
            box=grid.box,
            N=N,
            ), f, protocol=2)


if __name__ == '__main__':

    tests = [
        ((1, 1, 1), (8,8,128), 'axial2_box1'),
        ((1, 1, 2), (8,8,128), 'axial2_box2'),
        ((1, 1, 1.5), (8,8,64), 'axial1_box1.5'),
        ((1, 1, 1.5), (8,8,128), 'axial2_box1.5'),
        ((1, 1, 1.1), (8,8,64), 'axial1_box1.1'),
        ((1, 1, 1.9), (8,8,128), 'axial2_box1.9'),
        ((1, 1, 1.2), (8,8,64), 'axial1_box1.2'),
        ((1, 1, 1.8), (8,8,128), 'axial2_box1.8'),

        ((1, 1, 1), (16,8,64), 'radial2_box1'),
        ((2, 1, 1), (16,8,64), 'radial2_box2'),
        ((1.5, 1, 1), (8,8,64), 'radial1_box1.5'),
        ((1.5, 1, 1), (16,8,64), 'radial2_box1.5'),
        ((1.1, 1, 1), (8,8,64), 'radial1_box1.1'),
        ((1.9, 1, 1), (16,8,64), 'radial2_box1.9'),
        ((1.2, 1, 1), (8,8,64), 'radial1_box1.2'),
        ((1.8, 1, 1), (16,8,64), 'radial2_box1.8'),
        ]

    generate_gs(N=55000, shape=(8,8,64), box_modifiers=(1,1,1))
    for box_modifiers, shape, _ in tests:
        generate_gs(N=55000, shape=shape, box_modifiers=box_modifiers)
