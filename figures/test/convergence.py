from __future__ import print_function, division

import sys
import numpy
import pickle

import reikna.cluda as cluda

from beclab import *


class SZ2Sampler(Sampler):

    def __init__(self, wfs_meta):
        Sampler.__init__(self)
        self._grid = wfs_meta.grid

    def __call__(self, wfs_data, t):
        psi = wfs_data.get()
        return ((0.25 * (numpy.abs(psi[:,0]) ** 2 - numpy.abs(psi[:,1]) ** 2) ** 2).sum((1,2,3))
                * self._grid.dV)


def run_test(thr, stepper_cls, steps, wigner=False):

    print()
    print("*** Running " + stepper_cls.abbreviation + " test, " + str(steps) + " steps")
    print()

    # Simulation parameters
    trajectories = 64 if wigner else 1 # simulation trajectories
    interval = 0.1 # time interval
    samples = 1 # how many samples to take during simulation
    f_detuning = 37
    f_rabi = 350
    N = 55000
    state_dtype = numpy.complex128

    rng = numpy.random.RandomState(1234)

    freqs = (97.0, 97.0 * 1.03, 11.69)
    components = [const.rb87_1_minus1, const.rb87_2_1]
    scattering = const.scattering_matrix(components, B=const.magical_field_Rb87_1m1_2p1)
    losses = [
        (5.4e-42 / 6, (3, 0)),
        (8.1e-20 / 4, (0, 2)),
        (1.51e-20 / 2, (1, 1)),
        ]

    # Initial ground state
    with open('ground_states/ground_state_8-8-64_1-1-1.pickle', 'rb') as f:
        data = pickle.load(f)

    grid = UniformGrid(data['shape'], data['box'])
    potential = HarmonicPotential(freqs)
    system = System(components, scattering, potential=potential, losses=losses)

    psi = WavefunctionSet(thr, state_dtype, grid, components=2)
    psi.fill_with(data['data'])

    # Initial noise
    if wigner:
        psi = psi.to_wigner_coherent(trajectories, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(psi, f_detuning=f_detuning, f_rabi=f_rabi)

    integrator = Integrator(
        psi, system,
        trajectories=trajectories, stepper_cls=stepper_cls,
        wigner=wigner, seed=rng.randint(0, 2**32-1))

    # Integrate
    psi_sampler = PsiSampler()
    n_sampler = PopulationSampler(psi)
    sz2_sampler = SZ2Sampler(psi)

    bs(psi.data, 0, numpy.pi / 2)
    result, info = integrator.fixed_step(
        psi, 0, interval,
        steps, samples=samples,
        strong_convergence=['psi'],
        weak_convergence=['N', 'SZ2'],
        samplers=dict(psi=psi_sampler, N=n_sampler, SZ2=sz2_sampler)
        )

    return dict(
        weak_errors=info.weak_errors,
        strong_errors=info.strong_errors,
        t_integration=info.timings.integration,
        )


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    steppers = [
        CDIPStepper,
        CDStepper,
        RK4IPStepper,
        RK46NLStepper,
    ]

    import json
    import os.path

    step_nums = [2 ** pwr for pwr in range(10, 20)]

    wigner = bool(int(sys.argv[1]))

    if wigner:
        fname = 'convergence_wigner.json'
    else:
        fname = 'convergence_gpe.json'

    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            json.dump({}, f, indent=4)

    for stepper_cls in steppers:
        for steps in step_nums:
            result = run_test(thr, stepper_cls, steps, wigner=wigner)
            print(result)

            with open(fname, 'rb') as f:
                results = json.load(f)
            label = stepper_cls.abbreviation
            if label not in results:
                results[label] = {}
            results[label][steps] = result
            with open(fname, 'wb') as f:
                json.dump(results, f, indent=4)
