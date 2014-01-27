from __future__ import print_function, division

import sys
import numpy
import pickle

import reikna.cluda as cluda

from beclab import *
from beclab.integrator.results import sample


def calculate_ramsey(pulse_theta_noise=0, wigner=False, echo=False, t=1.0,
        steps=20000, samples=100, N=55000, trajectories=1, shape=(8, 8, 64),
        losses_enabled=True):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    f_detuning = 37
    f_rabi = 350
    state_dtype = numpy.complex128

    rng = numpy.random.RandomState(1234)

    freqs = (97.0, 97.0 * 1.03, 11.69)
    components = [const.rb87_1_minus1, const.rb87_2_1]

    scattering = const.scattering_3d(numpy.array([[100.4, 98.0], [98.0, 95.44]]), components[0].m)

    if losses_enabled:
        losses = [
            (5.4e-42 / 6, (3, 0)),
            (8.1e-20 / 4, (0, 2)),
            (1.51e-20 / 2, (1, 1)),
            ]
    else:
        losses = None

    potential = HarmonicPotential(freqs)
    system = System(components, scattering, potential=potential, losses=losses)

    # Initial state
    with open('ground_states/ground_state_8-8-64_1-1-1.pickle') as f:
        data = pickle.load(f)

    psi_gs = data['data']
    box = data['box']
    grid = UniformGrid(shape, box)
    print(grid.shape, grid.box)

    psi = WavefunctionSet(thr, numpy.complex128, grid, components=2)

    # Two-component state
    psi.fill_with(psi_gs)

    # Initial noise
    if wigner:
        psi = psi.to_wigner_coherent(trajectories, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(psi, f_detuning=f_detuning, f_rabi=f_rabi, seed=rng.randint(0, 2**32-1))

    integrator = Integrator(
        psi, system,
        trajectories=trajectories, stepper_cls=RK46NLStepper,
        wigner=wigner, seed=rng.randint(0, 2**32-1))

    # Integrate
    n_bs_sampler = PopulationSampler(psi, beam_splitter=bs, theta=numpy.pi/2)
    n_sampler = PopulationSampler(psi)
    i_sampler = InteractionSampler(psi)
    v_sampler = VisibilitySampler(psi)

    samplers = dict(N_bs=n_bs_sampler, N=n_sampler, V=v_sampler, I=i_sampler)
    weak_convergence = ['N', 'V', 'N_bs', 'I']

    bs(psi.data, 0, numpy.pi / 2, theta_noise=pulse_theta_noise)

    if t > 0:
        if echo:
            result1, info1 = integrator.fixed_step(
                psi, 0, t / 2, steps // 2, samples=samples // 2 if samples > 1 else 1,
                samplers=samplers, weak_convergence=weak_convergence)
            bs(psi.data, t / 2, numpy.pi, theta_noise=pulse_theta_noise)
            result2, info2 = integrator.fixed_step(
                psi, t / 2, t, steps // 2, samples=samples // 2 if samples > 1 else 1,
                samplers=samplers, weak_convergence=weak_convergence)

            result = concatenate_results(result1, result2)
            info = info2
        else:
            result, info = integrator.fixed_step(
                psi, 0, t, steps, samples=samples, samplers=samplers,
                weak_convergence=weak_convergence)
        weak_errors = info.weak_errors
    else:
        samples, _ = sample(psi.data, 0, samplers)
        result = {}
        for key in samples:
            result[key] = dict(
                trajectories=trajectories,
                time=numpy.array([0]))
            for subkey in ('mean', 'values', 'stderr'):
                if subkey in samples[key]:
                    result[key][subkey] = samples[key][subkey].reshape(1, *samples[key][subkey].shape)
        weak_errors = {key:0 for key in weak_convergence}

    psi_type = 'wigner' if wigner else 'wavefunction'

    return dict(result=result, weak_errors=weak_errors, psi_type=psi_type,
        N=N, steps=steps,
        shape=grid.shape, box=grid.box,
        wigner=wigner)


def concatenate_results(result1, result2):
    results = {}
    for key in result1:
        r = {}
        r1 = result1[key]
        r2 = result2[key]
        r['trajectories'] = r1['trajectories']
        for subkey in ('time', 'mean', 'values', 'stderr'):
            if subkey in r1:
                r[subkey] = numpy.concatenate([r1[subkey], r2[subkey]])
        results[key] = r
    return results


def calculate_echo(**kwds):
    kwds['echo'] = True
    total_steps = kwds['steps']
    total_samples = kwds['samples']
    total_t = float(kwds['t'])
    assert total_steps % total_samples == 0

    ress = None
    for j in xrange(total_samples + 1):
        steps = total_steps // total_samples * j
        t = total_t / total_samples * j
        print("--- Running Ramsey for t =", t, " steps =", steps)

        kwds['t'] = t
        kwds['steps'] = steps
        kwds['samples'] = 0
        res = calculate_ramsey(**kwds)
        if ress is None:
            ress = res
        else:
            for key in res['result']:
                ress['result'][key]['time'] = numpy.concatenate(
                    [ress['result'][key]['time'], numpy.array([res['result'][key]['time'][-1]])])
                for subkey in ('mean', 'values', 'stderr'):
                    if subkey in res['result'][key]:
                        ress['result'][key][subkey] = numpy.concatenate([
                            ress['result'][key][subkey],
                            res['result'][key][subkey][-1:]])
            ress['weak_errors'] = res['weak_errors']

    return ress


def run(func, fname, **kwds):
    data = func(**kwds)
    with open(fname, 'wb') as f:
        pickle.dump(data, f, protocol=2)


def test(short_time, test_name):

    ramsey_max_t = 1.3 if short_time else 3.0
    echo_max_t = 1.8 if short_time else 3.0
    steps = 80000 if short_time else 200000
    suffix = '' if short_time else '_long'
    fname = test_name + suffix + '.pickle'

    if test_name == 'ramsey_gpe_no_losses':
        run(calculate_ramsey, fname,
            losses_enabled=False,
            t=ramsey_max_t, steps=steps, samples=100, N=55000, wigner=False, trajectories=1, shape=(8,8,64))

    elif test_name == 'ramsey_gpe':
        run(calculate_ramsey, fname,
            t=ramsey_max_t, steps=steps, samples=100, N=55000, wigner=False, trajectories=1, shape=(8,8,64))
    elif test_name == 'echo_gpe':
        run(calculate_echo, fname,
            t=echo_max_t, steps=steps, samples=50, N=55000, wigner=False, trajectories=1, shape=(8,8,64))

    elif test_name == 'ramsey_wigner':
        run(calculate_ramsey, fname,
            t=ramsey_max_t, steps=steps, samples=100, N=55000, wigner=True, trajectories=128, shape=(8,8,64))
    elif test_name == 'echo_wigner':
        run(calculate_echo, fname,
            t=echo_max_t, steps=steps, samples=50, N=55000, wigner=True, trajectories=128, shape=(8,8,64))

    elif test_name == 'ramsey_wigner_varied_pulse':
        run(calculate_ramsey, fname,
            pulse_theta_noise=0.02,
            t=ramsey_max_t, steps=steps, samples=100, N=55000, wigner=True, trajectories=128, shape=(8,8,64))
    elif test_name == 'echo_wigner_varied_pulse':
        run(calculate_echo, fname,
            pulse_theta_noise=0.02,
            t=echo_max_t, steps=steps, samples=50, N=55000, wigner=True, trajectories=128, shape=(8,8,64))

    elif test_name == 'echo_wigner_single_run':
        run(calculate_ramsey, fname,
            echo=True,
            t=ramsey_max_t, steps=steps, samples=100, N=55000, wigner=True, trajectories=128, shape=(8,8,64))


if __name__ == '__main__':
    test(bool(int(sys.argv[1])), sys.argv[2])
