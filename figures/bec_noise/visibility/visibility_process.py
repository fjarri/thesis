import numpy
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from beclab import *
from beclab.meters import *


def get_visibility(fname):

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

    for t, psi in zip(times, psis):
        psi_gpu.fillWith(psi)
        psi_gpu.time = t
        ensembles = psi_gpu.ensembles

        Vs = im.getVisibilityPerEnsemble(psi_gpu)
        vis.append(Vs.mean())
        vis_errors.append(Vs.std() / numpy.sqrt(ensembles))

        #pulse.apply(psi_gpu, theta=0.5 * numpy.pi)
        Ns = psi_gpu.density_meter.getNPerEnsemble().get()
        N1.append(Ns[0].mean())
        N2.append(Ns[1].mean())
        N1_errors.append(Ns[0].std() / numpy.sqrt(ensembles))
        N2_errors.append(Ns[1].std() / numpy.sqrt(ensembles))

    env.release()
    return dict(
        times=list(times),
        visibility=vis,
        visibility_errors=vis_errors,
        N1=N1, N2=N2,
        N1_errors=N1_errors, N2_errors=N2_errors)

outputs = [
    ('ramsey_gpe.pickle', 'ramsey_gpe_vis.json'),
    ('echo_gpe.pickle', 'echo_gpe_vis.json'),
    ('ramsey_wigner.pickle', 'ramsey_wigner_vis.json'),
    ('echo_wigner.pickle', 'echo_wigner_vis.json'),
    ('ramsey_long_gpe.pickle', 'ramsey_long_gpe_vis.json'),
    ('echo_long_gpe.pickle', 'echo_long_gpe_vis.json'),
    ('ramsey_long_wigner.pickle', 'ramsey_long_wigner_vis.json'),
    ('echo_long_wigner.pickle', 'echo_long_wigner_vis.json'),
]

for f_pickle, f_json in outputs:
    with open('visibility/' + f_json, 'w') as f:
        json.dump(get_visibility(f_pickle), f, indent=4)
