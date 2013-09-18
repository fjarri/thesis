import numpy
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from beclab import *
from beclab.meters import *


def get_phnoise(fname):

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

    print fname, "ensembles =", psi_gpu.ensembles
    unc = UncertaintyMeter(env, constants, grid)

    ph_noises = []
    pz_noises = []

    for t, psi in zip(times, psis):
        psi_gpu.fillWith(psi)
        pz_noises.append(unc.getPzNoise(psi_gpu))
        ph_noises.append(unc.getPhaseNoise(psi_gpu))

    env.release()
    return dict(times=list(times), phnoise=ph_noises, pznoise=pz_noises)


def simulate_phnoise(fname, approx_t, echo=True):

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

            phis_expected = numpy.random.rand(ensembles) * 2 * numpy.pi
            MW_noise = 0.125 if echo else 0.5
            phis = numpy.random.normal(scale=MW_noise * t, size=ensembles) + phis_expected
            thetas = numpy.random.normal(loc=numpy.pi / 2, scale=0.02, size=ensembles)

            # rotation
            c = numpy.cos(thetas / 2)
            s = numpy.sin(thetas / 2)
            tt = numpy.exp(1j * phis)

            Ns0 = (c ** 2 * Ns[0] + s ** 2 * Ns[1]
                - 1j * tt.conj() * c * s * Is + 1j * tt * c * s * Is.conj()).real
            Ns1 = (s ** 2 * Ns[0] + c ** 2 * Ns[1]
                + 1j * tt.conj() * c * s * Is - 1j * tt * c * s * Is.conj()).real

            Ns0 *= numpy.random.normal(loc=1, scale=0.023, size=ensembles)
            Ns1 *= numpy.random.normal(loc=1, scale=0.023, size=ensembles)

            Pz = (Ns1 - Ns0) / (Ns0 + Ns1)

            return dict(time=t, Pz=list(Pz), phis=list(phis_expected))


outputs = [
    ('ramsey_wigner.pickle', 'ramsey_wigner_noise.json'),
    ('echo_wigner.pickle', 'echo_wigner_noise.json'),
    ('ramsey_wigner_varied_pulse.pickle', 'ramsey_wigner_varied_pulse_noise.json'),
    ('echo_wigner_varied_pulse.pickle', 'echo_wigner_varied_pulse_noise.json'),
    ('echo_wigner_varied_first_pulse.pickle', 'echo_wigner_varied_first_pulse_noise.json'),
]

for f_pickle, f_json in outputs:
    with open('noise/' + f_json, 'w') as f:
        json.dump(get_phnoise(f_pickle), f, indent=4)


sim_phnoise = [
    ('ramsey_wigner.pickle', 0.02, 'ramsey_sim_phnoise_20ms.json'),
    ('ramsey_wigner.pickle', 0.45, 'ramsey_sim_phnoise_450ms.json')]

for f_pickle, t, f_json in sim_phnoise:
    with open('noise/' + f_json, 'w') as f:
        json.dump(simulate_phnoise(f_pickle, t), f, indent=4)
