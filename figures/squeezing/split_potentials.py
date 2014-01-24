"""
This example demonstrates usage of static per-component potentials
"""

import pickle
import numpy
import sys

from reikna import cluda
from beclab import *


# parameters from Riedel at al. (2010)
N = 1250
freqs = (500, 500, 109)
f_rabi = 2100
f_detuning = -40
potentials_separation = 0.52e-6
splitting_time = 12.7e-3
shape = (16, 16, 128)
state_dtype = numpy.complex128
steps = 20000
samples = 100


components = [const.rb87_1_minus1, const.rb87_2_1]
scattering = const.scattering_3d(
    numpy.array([[100.4, 97.7], [97.7, 95.0]]), components[0].m)

losses = [
    (5.4e-42 / 6, (3, 0)),
    (8.1e-20 / 4, (0, 2)),
    (1.51e-20 / 2, (1, 1)),
    ]

potential_init = HarmonicPotential(freqs)
potential_split = HarmonicPotential(
    freqs,
    displacements=[
        (0, 2, -potentials_separation / 2),
        (1, 2, potentials_separation / 2)])
system_init = System(components, scattering, potential=potential_init, losses=losses)
system_split = System(components, scattering, potential=potential_split, losses=losses)

box = box_for_tf(system_init, 0, N)
box = (box[0], box[1], box[2] + potentials_separation * 2)
grid = UniformGrid(shape, box)

cutoff = WavelengthCutoff.for_energy(5000 * const.HBAR, components[0])
print("Using", cutoff.get_modes_number(grid), "modes out of", grid.size)


def run_pass(trajectories=128):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    rng = numpy.random.RandomState(1234)

    gs_gen = ImaginaryTimeGroundState(thr, state_dtype, grid, system_init,
        stepper_cls=RK46NLStepper, cutoff=cutoff)

    # Ground state
    psi = gs_gen([N, 0], E_diff=1e-7, E_conv=1e-9, sample_time=1e-6)

    # Initial noise
    psi = psi.to_wigner_coherent(trajectories, seed=rng.randint(0, 2**32-1))

    integrator = Integrator(
        psi, system_split,
        trajectories=trajectories,
        stepper_cls=RK46NLStepper, cutoff=cutoff,
        wigner=True, seed=rng.randint(0, 2**32-1))

    # Prepare samplers
    bs = BeamSplitter(psi, f_detuning=f_detuning, f_rabi=f_rabi)
    n_bs_sampler = PopulationSampler(psi, beam_splitter=bs, theta=numpy.pi / 2)
    n_sampler = PopulationSampler(psi)
    i_sampler = InteractionSampler(psi)
    v_sampler = VisibilitySampler(psi)
    v_sampler.no_values = True
    ax_sampler = Density1DSampler(psi, axis=2)
    ax_sampler.no_values = True

    samplers = dict(
        N=n_sampler, I=i_sampler, V=v_sampler,
        N_bs=n_bs_sampler, axial_density=ax_sampler)

    # Integrate
    bs(psi.data, 0, numpy.pi / 2)

    result, info = integrator.fixed_step(
        psi, 0, splitting_time, steps, samples=samples,
        samplers=samplers, weak_convergence=['N', 'I', 'V'])

    return result, info


def combined_test(fname, total_trajectories):

    chunk = 128

    full_results = None

    for i in range(total_trajectories // chunk):

        print("Chunk", i)

        result, info = run_pass(trajectories=chunk)

        if full_results is None:
            full_results = result
        else:
            full_results = join_results(full_results, result)

        with open(fname, 'wb') as f:
            pickle.dump(dict(
                errors=info.weak_errors,
                results=full_results), f, protocol=2)


if __name__ == '__main__':
    combined_test('split_potentials.pickle', 128 * 100)
