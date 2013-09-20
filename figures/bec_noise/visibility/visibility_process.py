import numpy
import pickle
import json
from scipy.optimize import leastsq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from beclab import *
from beclab.meters import *


def fit_visibility(t, Ns, Is, echo=False):

    ensembles = Is.size

    MW_noise = (0.125 if echo else 0.5) * t # in radians
    imaging_noise = 0.023 # in percent
    theta_noise = 0.02 # in radians

    phis = numpy.random.rand(ensembles) * 2 * numpy.pi
    #phis = numpy.linspace(numpy.pi / 8, numpy.pi - numpy.pi / 8, ensembles)
    if MW_noise > 0:
        phis_real = numpy.random.normal(scale=MW_noise, size=ensembles) + phis
    else:
        phis_real = phis
    thetas = numpy.random.normal(loc=numpy.pi / 2, scale=theta_noise, size=ensembles)

    # rotation
    c = numpy.cos(thetas / 2)
    s = numpy.sin(thetas / 2)
    tt = numpy.exp(1j * phis_real)

    Ns0 = (c ** 2 * Ns[0] + s ** 2 * Ns[1]
        - 1j * tt.conj() * c * s * Is + 1j * tt * c * s * Is.conj()).real
    Ns1 = (s ** 2 * Ns[0] + c ** 2 * Ns[1]
        + 1j * tt.conj() * c * s * Is - 1j * tt * c * s * Is.conj()).real

    Ns0 *= numpy.random.normal(loc=1, scale=0.023, size=ensembles)
    Ns1 *= numpy.random.normal(loc=1, scale=0.023, size=ensembles)

    Pz = (Ns1 - Ns0) / (Ns0 + Ns1)

    # sort phis
    pairs = [[phi, pz] for phi, pz in zip(phis, Pz)]
    pairs = numpy.array(sorted(pairs, key=lambda x: x[0])).T
    phis = pairs[0]
    Pz = pairs[1]

    # Fitting with a cosine function
    guess_amp = 3 * numpy.std(Pz) / (2**0.5)
    guess_phase = 0
    optimize_func = lambda x: x[0] * numpy.cos(phis + x[1]) - Pz
    est_amp, est_phase = leastsq(optimize_func, [guess_amp, guess_phase])[0]

    # Find the phase noise
    diffs = []
    for phi, pz in zip(phis, Pz):
        pz /= est_amp
        if abs(pz) > 1:
            continue

        g1 = (numpy.arccos(pz) - est_phase) % (2 * numpy.pi)
        g2 = (-numpy.arccos(pz) - est_phase) % (2 * numpy.pi)

        def circular_diff(a1, a2):
            if abs(a1 - a2) < numpy.pi:
                return a1 - a2
            elif a1 < a2:
                return 2 * numpy.pi + a1 - a2
            else:
                return a1 - 2 * numpy.pi - a2

        d1 = circular_diff(g1, phi)
        d2 = circular_diff(g2, phi)

        if abs(d1) < abs(d2):
            diffs.append(d1)
        else:
            diffs.append(d2)

    diffs = numpy.array(diffs)
    est_phnoise = diffs.std()

    return abs(est_amp), est_phnoise, \
        dict(phis=list(phis), Pz=list(Pz), est_phnoise=est_phnoise,
            est_amp=est_amp, est_phase=est_phase)



def get_visibility(fname, echo=False, technical_noises=False):

    with open(fname, 'rb') as f:
        res = pickle.load(f)

    env = envs.cuda(device_num=0)
    constants_kwds = res['constants_kwds']
    constants = Constants(double=env.supportsDouble(), **constants_kwds)
    grid = UniformGrid.forN(env, constants, res['N'], res['shape'])

    times = res['times']
    psis = res['psis']

    psi_gpu = WavefunctionSet(env, constants, grid,
        ensembles=psis[0].shape[1], components=psis[0].shape[0], psi_type=res['psi_type'])

    print fname, "ensembles =", psi_gpu.ensembles
    im = IntegralMeter(env, constants, grid)
    pulse = Pulse(env, constants, grid, f_rabi=350)

    vis = []
    vis_errors = []
    N1 = []
    N2 = []
    N1_errors = []
    N2_errors = []
    est_vis = []
    est_phnoises = []

    for t, psi in zip(times, psis):
        psi_gpu.fillWith(psi)
        psi_gpu.time = t
        ensembles = psi_gpu.ensembles

        Vs = im.getVisibilityPerEnsemble(psi_gpu)
        vis.append(Vs.mean())
        vis_errors.append(Vs.std() / numpy.sqrt(ensembles))

        #pulse.apply(psi_gpu, theta=0.5 * numpy.pi)
        Ns = psi_gpu.density_meter.getNPerEnsemble().get()
        Is = psi_gpu.interaction_meter.getIPerEnsemble().get()

        N1.append(Ns[0].mean())
        N2.append(Ns[1].mean())
        N1_errors.append(Ns[0].std() / numpy.sqrt(ensembles))
        N2_errors.append(Ns[1].std() / numpy.sqrt(ensembles))

        if technical_noises:
            Ns = numpy.hstack([Ns] * 20)
            Is = numpy.hstack([Is] * 20)

            est_V, est_phnoise, _ = fit_visibility(t, Ns, Is, echo=echo)
            est_vis.append(est_V)
            est_phnoises.append(est_phnoise)


    env.release()
    return dict(
        times=list(times),
        visibility=vis,
        visibility_errors=vis_errors,
        N1=N1, N2=N2,
        N1_errors=N1_errors, N2_errors=N2_errors,
        est_visibility=est_vis,
        est_phnoises=est_phnoises)


def simulate_phnoise(fname, approx_t, echo=False):

    with open(fname, 'rb') as f:
        res = pickle.load(f)

    env = envs.cuda(device_num=1)
    constants_kwds = res['constants_kwds']
    constants = Constants(double=env.supportsDouble(), **constants_kwds)
    grid = UniformGrid.forN(env, constants, res['N'], res['shape'])

    times = res['times']
    psis = res['psis']

    psi_gpu = WavefunctionSet(env, constants, grid,
        ensembles=psis[0].shape[1], components=psis[0].shape[0], psi_type=res['psi_type'])

    for t, psi in zip(times, psis):
        if t > approx_t:

            psi_gpu.fillWith(psi)
            psi_gpu.time = t
            ensembles = psi_gpu.ensembles

            Ns = psi_gpu.density_meter.getNPerEnsemble().get()
            Is = psi_gpu.interaction_meter.getIPerEnsemble().get()
            env.release()

            _, _, pz_points = fit_visibility(t, Ns, Is, echo=echo)
            pz_points['time'] = t

            return pz_points



outputs = [
    ('ramsey_gpe_no_losses.pickle', 'ramsey_gpe_no_losses_vis.json'),
    ('ramsey_gpe.pickle', 'ramsey_gpe_vis.json'),
    ('echo_gpe.pickle', 'echo_gpe_vis.json'),

    ('ramsey_wigner.pickle', 'ramsey_wigner_vis.json'),
    ('echo_wigner.pickle', 'echo_wigner_vis.json'),
    ('ramsey_wigner_varied_pulse.pickle', 'ramsey_wigner_varied_pulse_vis.json', False, True),
    ('echo_wigner_varied_pulse.pickle', 'echo_wigner_varied_pulse_vis.json', True, True),
    ('ramsey_long_wigner_varied_pulse.pickle', 'ramsey_long_wigner_varied_pulse_vis.json', False, True),
    ('echo_long_wigner_varied_pulse.pickle', 'echo_long_wigner_varied_pulse_vis.json', True, True),

    ('ramsey_long_gpe.pickle', 'ramsey_long_gpe_vis.json'),
    ('echo_long_gpe.pickle', 'echo_long_gpe_vis.json'),
    ('ramsey_long_wigner.pickle', 'ramsey_long_wigner_vis.json'),
    ('echo_long_wigner.pickle', 'echo_long_wigner_vis.json'),
    ('echo_wigner_single_run.pickle', 'echo_wigner_single_run.json'),
]

for args in outputs:
    if len(args) == 2:
        f_pickle, f_json = args
        echo = False
        technical_noises = False
    else:
        f_pickle, f_json, echo, technical_noises = args
    with open('visibility/' + f_json, 'w') as f:
        json.dump(get_visibility(f_pickle, echo=echo, technical_noises=technical_noises), f, indent=4)


sim_phnoise = [
    ('ramsey_wigner.pickle', 0.02, 'ramsey_sim_phnoise_20ms.json'),
    ('ramsey_wigner.pickle', 0.45, 'ramsey_sim_phnoise_450ms.json')]

for f_pickle, t, f_json in sim_phnoise:
    with open('noise/' + f_json, 'w') as f:
        json.dump(simulate_phnoise(f_pickle, t), f, indent=4)
