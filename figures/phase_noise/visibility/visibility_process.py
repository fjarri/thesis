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

    vis = []
    vis_errors = []

    for t, psi in zip(times, psis):
        psi_gpu.fillWith(psi)
        Vs = im.getVisibilityPerEnsemble(psi_gpu)
        vis.append(Vs.mean())
        vis_errors.append(Vs.std() / numpy.sqrt(float(Vs.size)))

    env.release()
    return numpy.array(res['times']), numpy.array(vis), numpy.array(vis_errors)


t_r_gpe, v_r_gpe, _ = get_visibility('ramsey_gpe.pickle')
t_e_gpe, v_e_gpe, _ = get_visibility('echo_gpe.pickle')
t_r_wig, v_r_wig, ve_r_wig = get_visibility('ramsey_wigner.pickle')
t_e_wig, v_e_wig, ve_e_wig = get_visibility('echo_wigner.pickle')

with open('ramsey_gpe_vis.json', 'w') as f:
    json.dump(dict(times=t_r_gpe.tolist(), visibility=v_r_gpe.tolist()), f)
with open('ramsey_wigner_vis.json', 'w') as f:
    json.dump(dict(times=t_r_wig.tolist(), visibility=v_r_wig.tolist(),
        visibility_errors=ve_r_wig.tolist()), f)
with open('echo_gpe_vis.json', 'w') as f:
    json.dump(dict(times=t_e_gpe.tolist(), visibility=v_e_gpe.tolist()), f)
with open('echo_wigner_vis.json', 'w') as f:
    json.dump(dict(times=t_e_wig.tolist(), visibility=v_e_wig.tolist(),
        visibility_errors=ve_e_wig.tolist()), f)


t_r_gpe, v_r_gpe, _ = get_visibility('ramsey_long_gpe.pickle')
t_e_gpe, v_e_gpe, _ = get_visibility('echo_long_gpe.pickle')
t_r_wig, v_r_wig, ve_r_wig = get_visibility('ramsey_long_wigner.pickle')
t_e_wig, v_e_wig, ve_e_wig = get_visibility('echo_long_wigner.pickle')

with open('ramsey_long_gpe_vis.json', 'w') as f:
    json.dump(dict(times=t_r_gpe.tolist(), visibility=v_r_gpe.tolist()), f)
with open('ramsey_long_wigner_vis.json', 'w') as f:
    json.dump(dict(times=t_r_wig.tolist(), visibility=v_r_wig.tolist(),
        visibility_errors=ve_r_wig.tolist()), f)
with open('echo_long_gpe_vis.json', 'w') as f:
    json.dump(dict(times=t_e_gpe.tolist(), visibility=v_e_gpe.tolist()), f)
with open('echo_long_wigner_vis.json', 'w') as f:
    json.dump(dict(times=t_e_wig.tolist(), visibility=v_e_wig.tolist(),
        visibility_errors=ve_e_wig.tolist()), f)

