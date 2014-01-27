"""
Check squeezing for different a12 values near Feschbach resonance.
Data is accumulated over several iterations which helps if many trajectories
are necessary and there is not enough GPU memory to process them at once.
"""

import sys
import numpy
import json, pickle

import reikna.cluda.dtypes as dtypes
import reikna.cluda as cluda
from beclab import *


lattice_size = (16, 16, 128) # spatial lattice points
interval = 0.1 # time interval
samples = 200 # how many samples to take during simulation
steps = samples * 100 # number of time steps (should be multiple of samples)
f_detuning = 37
f_rabi = 350
N = 55000
state_dtype = numpy.complex128
freqs = (97.0, 97.0 * 1.03, 11.69)
components = [const.rb87_1_1, const.rb87_2_minus1]

params = [
    (80.0, 38.5e-19),
    (85.0, 19.3e-19),
    (90.0, 7.00e-19),
    (95.0, 0.853e-19)
]

losses_enabled = bool(int(sys.argv[1]))
param_idx = int(sys.argv[2])

if sys.argv[3] == 'cd':
    stepper_cls = CDStepper
elif sys.argv[3] == 'rk46nl':
    stepper_cls = RK46NLStepper

a12, gamma12 = params[param_idx]
fname = 'feshbach_a12_' + str(a12) + ("" if losses_enabled else "_no_losses") + '.pickle'

scattering = const.scattering_3d(
    numpy.array([[100.4, a12], [a12, 95.44]]), components[0].m)

if losses_enabled:
    losses = [
        (gamma12 / 2, (1, 1)),
        ]
else:
    losses = None

potential = HarmonicPotential(freqs)
system = System(components, scattering, potential=potential, losses=losses)
grid = UniformGrid(lattice_size, box_for_tf(system, 0, N))
#cutoff = WavelengthCutoff.padded(grid, pad=4)
cutoff = None


def test_uncertainties(trajectories=128):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    rng = numpy.random.RandomState(1234)

    gs_gen = ImaginaryTimeGroundState(thr, state_dtype, grid, system, cutoff=cutoff)

    # Ground state
    psi = gs_gen([N, 0], E_diff=1e-7, E_conv=1e-9, sample_time=1e-6)

    # Initial noise
    psi = psi.to_wigner_coherent(trajectories, seed=rng.randint(0, 2**32-1))

    integrator = Integrator(
        psi, system,
        trajectories=trajectories, stepper_cls=stepper_cls,
        wigner=True, seed=rng.randint(0, 2**32-1),
        cutoff=cutoff)

    # Prepare samplers
    bs = BeamSplitter(psi, f_detuning=f_detuning, f_rabi=f_rabi)
    n_sampler = PopulationSampler(psi)
    i_sampler = InteractionSampler(psi)
    samplers = dict(N=n_sampler, I=i_sampler)

    # Integrate
    bs(psi.data, 0, numpy.pi / 2)
    result, info = integrator.fixed_step(
        psi, 0, interval, steps, samples=samples,
        samplers=samplers, weak_convergence=['N', 'I'])

    return result, info


def combined_test(fname, total_trajectories):

    chunk = 256

    full_results = None

    for i in range(total_trajectories // chunk):

        print("Chunk", i)

        result, info = test_uncertainties(trajectories=chunk)

        if full_results is None:
            full_results = result
        else:
            full_results = join_results(full_results, result)

        with open(fname, 'wb') as f:
            pickle.dump(dict(
                a12=a12, gamma12=gamma12, losses_enabled=losses_enabled,
                errors=info.weak_errors,
                results=full_results), f, protocol=2)


if __name__ == '__main__':
    combined_test(fname, 2560)
