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
    return numpy.array(res['times']), numpy.array(ph_noises), numpy.array(pz_noises)


t_r_wig, ph_r_wig, pz_r_wig = get_phnoise('ramsey_wigner.pickle')
t_e_wig, ph_e_wig, pz_e_wig = get_phnoise('echo_wigner.pickle')

with open('ramsey_wigner_noise.json', 'w') as f:
    json.dump(dict(times=t_r_wig.tolist(), phnoise=ph_r_wig.tolist(),
        pznoise=pz_r_wig.tolist()), f)
with open('echo_wigner_noise.json', 'w') as f:
    json.dump(dict(times=t_e_wig.tolist(), phnoise=ph_e_wig.tolist(),
        pznoise=pz_e_wig.tolist()), f)

t_r_wig, ph_r_wig, pz_r_wig = get_phnoise('ramsey_wigner_varied_pulse.pickle')
t_e_wig, ph_e_wig, pz_e_wig = get_phnoise('echo_wigner_varied_pulse.pickle')

with open('ramsey_wigner_varied_pulse_noise.json', 'w') as f:
    json.dump(dict(times=t_r_wig.tolist(), phnoise=ph_r_wig.tolist(),
        pznoise=pz_r_wig.tolist()), f)
with open('echo_wigner_noise.json', 'w') as f:
    json.dump(dict(times=t_e_wig.tolist(), phnoise=ph_e_wig.tolist(),
        pznoise=pz_e_wig.tolist()), f)

