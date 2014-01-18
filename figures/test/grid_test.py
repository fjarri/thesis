from __future__ import print_function, division

import sys
import numpy
import pickle

import reikna.cluda as cluda

from beclab import *


def calculate_ramsey(wigner=False, t=1.0,
        steps=20000, samples=100, N=55000, trajectories=1, shape=(8, 8, 64),
        losses=True, linear_losses=None, box_modifiers=None):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    f_detuning = 37
    f_rabi = 350
    state_dtype = numpy.complex128

    rng = numpy.random.RandomState(1234)

    freqs = (97.0, 97.0 * 1.03, 11.69)
    components = [const.rb87_1_minus1, const.rb87_2_1]

    scattering = const.scattering_3d(numpy.array([[100.4, 98.0], [98.0, 95.44]]), components[0].m)

    if linear_losses is not None:
        losses = [
            (linear_losses, (1, 0)),
            (linear_losses, (0, 1)),
            ]
    elif losses:
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
    if box_modifiers is None:
        box_modifiers = (1, 1, 1)
    with open('ground_states/ground_state_'
            + '-'.join([str(s) for s in shape])
            + '_'
            + '-'.join([str(b) for b in box_modifiers]) + '.pickle', 'rb') as f:
        data = pickle.load(f)

    psi_gs = data['data']
    box = data['box']
    grid = UniformGrid(shape, box)
    print(grid.shape, box_modifiers, grid.box)

    psi = WavefunctionSet(thr, numpy.complex128, grid, components=2)

    # Two-component state
    psi.fill_with(psi_gs)

    # Initial noise
    if wigner:
        psi = psi.to_wigner_coherent(trajectories, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(psi, f_detuning=f_detuning, f_rabi=f_rabi)

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

    bs(psi.data, 0, numpy.pi / 2)

    if t > 0:
        result, info = integrator.fixed_step(
            psi, 0, t, steps, samples=samples, samplers=samplers,
            weak_convergence=weak_convergence)
    else:
        info = None

    psi_type = 'wigner' if wigner else 'wavefunction'

    return dict(result=result, weak_errors=info.weak_errors, psi_type=psi_type,
        N=N, steps=steps,
        shape=grid.shape, box=grid.box,
        wigner=wigner)


def join_results(results1, results2):
    assert set(results1.keys()) == set(results2.keys())
    full_results = {}
    for key in results1:
        r1 = results1[key]
        r2 = results2[key]
        tr1 = r1['trajectories']
        tr2 = r2['trajectories']

        assert all(r1['time'] == r2['time'])
        full_results[key] = dict(time=r1['time'], trajectories=tr1 + tr2)

        if 'values' in r1:
            full_results[key]['values'] = numpy.concatenate([r1['values'], r2['values']], axis=1)
        if 'mean' in r1:
            mean1 = r1['mean']
            mean2 = r2['mean']
            full_results[key]['mean'] = (mean1 * tr1 + mean2 * tr2) / (tr1 + tr2)
        if 'stderr' in r1:
            err1 = r1['stderr']
            err2 = r2['stderr']
            full_results[key]['stderr'] = numpy.sqrt(
                (err1**2 * tr1**2 + err2**2 * tr2**2)) / (tr1 + tr2)

    return full_results


def run(func, fname, ens_step, **kwds):

    total_trajectories = kwds['trajectories']
    kwds['trajectories'] = ens_step

    for j in xrange(0, total_trajectories, ens_step):
        print("*** Ensembles:", j, "to", j + ens_step - 1)
        data = func(**kwds)

        if j > 0:
            with open(fname, 'rb') as f:
                data_full = pickle.load(f)

            result_full = data_full['result']
            result_part = data['result']

            data['result'] = join_results(result_full, result_part)

        with open(fname, 'wb') as f:
            pickle.dump(data, f, protocol=2)


def spatial_convergence(wigner, start_num, end_num):

    t_grid = 1.3
    steps_grid = 160000
    ens_grid = 64 if wigner else 1

    prefix = 'ramsey_' + ('wigner' if wigner else 'gpe') + '_test'

    tests = [
        ((1, 1, 1), (8,8,64), None),

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

    for box_modifiers, shape, suffix in tests[start_num:end_num+1]:
        run(
            calculate_ramsey,
            prefix + ('_' + suffix if suffix is not None else '') + '.pickle', ens_grid,
            box_modifiers=box_modifiers,
            t=t_grid, steps=steps_grid, samples=100, N=55000, wigner=wigner,
            trajectories=ens_grid, shape=shape)


if __name__ == '__main__':

    wigner = bool(int(sys.argv[1]))
    test_start = int(sys.argv[2])
    test_end = int(sys.argv[3])

    spatial_convergence(wigner, test_start, test_end)
