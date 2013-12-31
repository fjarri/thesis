from __future__ import print_function, division

import numpy
import pickle

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module

from beclab.integrator import (
    Sampler, Integrator, Wiener, Drift, Diffusion,
    SSCDStepper, CDStepper, RK4IPStepper, RK46NLStepper)
from beclab.modules import get_drift, get_diffusion
from beclab.grid import UniformGrid, box_3D
from beclab.beam_splitter import BeamSplitter
import beclab.constants as const
from beclab.ground_state import get_TF_state


class PsiSampler(Sampler):

    def __init__(self):
        Sampler.__init__(self, no_average=True)

    def __call__(self, psi, t):
        return psi.get().transpose(1, 0, *range(2, len(psi.shape)))


class NSampler(Sampler):

    def __init__(self, thr, grid, psi_temp, beam_splitter, wigner=False):
        Sampler.__init__(self)
        self._thr = thr
        self._wigner = wigner
        self._grid = grid
        self._beam_splitter = beam_splitter
        self._psi_temp = psi_temp

    def __call__(self, psi, t):
        self._thr.copy_array(psi, dest=self._psi_temp)
        self._beam_splitter(self._psi_temp, t, numpy.pi / 2)
        psi = self._psi_temp.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)
        return (density.sum((2, 3, 4)) * self._grid.dV).T


class SZ2Sampler(Sampler):

    def __init__(self, thr, grid, psi_temp, beam_splitter, wigner=False):
        Sampler.__init__(self)
        self._thr = thr
        self._wigner = wigner
        self._grid = grid
        self._beam_splitter = beam_splitter
        self._psi_temp = psi_temp

    def __call__(self, psi, t):
        self._thr.copy_array(psi, dest=self._psi_temp)
        self._beam_splitter(self._psi_temp, t, numpy.pi / 2)
        psi = self._psi_temp.get()

        return ((0.25 * (numpy.abs(psi[0]) ** 2 - numpy.abs(psi[1]) ** 2) ** 2).sum((1,2,3))
                * self._grid.dV)


def run_test(thr, stepper_cls, steps, wigner=False):

    print()
    print("*** Running " + stepper_cls.abbreviation + " test, " + str(steps) + " steps")
    print()

    # Simulation parameters
    lattice_size = (8, 8, 64) # spatial lattice points
    paths = 64 # simulation paths
    interval = 0.1 # time interval
    samples = 1 # how many samples to take during simulation
    #steps = samples * 50 # number of time steps (should be multiple of samples)
    gamma = 0.2
    f_detuning = 37
    f_rabi = 350
    N = 55000
    state_dtype = numpy.complex128

    rng = numpy.random.RandomState(1234)

    freqs = (97.6, 97.6, 11.96)
    states = [const.Rb87_1_minus1, const.Rb87_2_1]

    get_g = lambda s1, s2: const.get_scattering_constant(
        const.get_interaction_constants(const.B_Rb87_1m1_2p1, s1, s2)[0], s1.m)
    scattering = numpy.array([[get_g(s1, s2) for s2 in states] for s1 in states])
    losses = [
        (5.4e-42 / 6, (3, 0)),
        (8.1e-20 / 4, (0, 2)),
        (1.51e-20 / 2, (1, 1)),
        ]

    grid = UniformGrid(lattice_size, box_3D(N, freqs, states[0]))

    # Initial TF state
    psi = get_TF_state(thr, grid, state_dtype, states, freqs, [N, 0])

    # Initial ground state
    with open('ground_state_8-8-64_1-1-1.pickle', 'rb') as f:
        data = pickle.load(f)

    psi_gs = data['data']
    psi.fill_with(psi_gs)

    # Initial noise
    if wigner:
        psi = psi.to_wigner_coherent(paths, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(thr, psi.data, 0, 1, f_detuning=f_detuning, f_rabi=f_rabi)

    # Prepare integrator components
    drift = get_drift(state_dtype, grid, states, freqs, scattering, losses, wigner=wigner)
    diffusion = get_diffusion(state_dtype, grid, 2, losses) if wigner else None

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=1j * const.HBAR / (2 * states[0].m),
        ensembles=paths if wigner else 1,
        diffusion=diffusion)

    if wigner:
        wiener = Wiener(stepper.parameter.dW, 1. / grid.dV, seed=rng.randint(0, 2**32-1))
    else:
        wiener = None
    integrator = Integrator(thr, stepper, wiener=wiener, profile=True)

    # Integrate
    psi_temp = thr.empty_like(psi.data)
    psi_sampler = PsiSampler()
    n_sampler = NSampler(thr, grid, psi_temp, bs, wigner=wigner)
    sz2_sampler = SZ2Sampler(thr, grid, psi_temp, bs, wigner=wigner)

    bs(psi.data, 0, numpy.pi / 2)
    result, info = integrator.fixed_step(
        psi.data, 0, interval, steps, samples=samples,
        convergence=['psi', 'N', 'SZ2'],
        samplers=dict(psi=psi_sampler, N=n_sampler, SZ2=sz2_sampler))

    N_mean = result['N']
    N_exact = N * numpy.exp(-gamma * result['time'] * 2)

    return dict(
        errors=info.errors,
        N_diff=abs(N_mean.sum(1)[-1] - N_exact[-1]) / N_exact[-1],
        t_integration=info.timings.integration,
        )


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    steppers = [
        SSCDStepper,
        CDStepper,
        RK4IPStepper,
        RK46NLStepper,
    ]

    import json
    import os.path

    step_nums = [2 ** pwr for pwr in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]

    for wigner in (False, True):
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
