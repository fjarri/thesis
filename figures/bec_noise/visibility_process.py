import numpy
import pickle
import json

from scipy.optimize import leastsq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit_visibility(t, Ns, Is, echo=False, tech_noise=False):

    trajectories = Is.size

    if tech_noise:
        MW_noise = (0.125 if echo else 0.5) * t # in radians
        imaging_noise = 0.023 # in percent
        theta_noise = 0.02 # in radians
    else:
        MW_noise = 0
        imaging_noise = 0
        theta_noise = 0

    phis = numpy.random.rand(trajectories) * 2 * numpy.pi
    phis_real = numpy.random.normal(size=trajectories) * MW_noise + phis
    thetas = numpy.random.normal(size=trajectories) * theta_noise + numpy.pi / 2

    # rotation
    c = numpy.cos(thetas / 2)
    s = numpy.sin(thetas / 2)
    tt = numpy.exp(1j * phis_real)

    Ns0 = (c ** 2 * Ns[:,0] + s ** 2 * Ns[:,1]
        - 1j * tt.conj() * c * s * Is + 1j * tt * c * s * Is.conj()).real
    Ns1 = (s ** 2 * Ns[:,0] + c ** 2 * Ns[:,1]
        + 1j * tt.conj() * c * s * Is - 1j * tt * c * s * Is.conj()).real

    Ns0 *= (1 + numpy.random.normal(size=trajectories) * imaging_noise)
    Ns1 *= (1 + numpy.random.normal(size=trajectories) * imaging_noise)

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

    return dict(
        phis=list(phis), Pz=list(Pz), phnoise=est_phnoise,
        amplitude=est_amp, phase=est_phase)


def get_pz(fname, t_sample):
    with open(fname + '.pickle', 'rb') as f:
        data = pickle.load(f)

    result = data['result']
    echo = 'echo' in fname
    tech_noise = 'varied_pulse' in fname

    for i in range(len(result['N']['time'])):

        t = result['N']['time'][i]
        if t > t_sample:
            Ns = result['N']['values'][i]
            Is = result['I']['values'][i]

            Ns = numpy.concatenate([Ns], axis=0)
            Is = numpy.concatenate([Is])

            est_results = fit_visibility(t, Ns, Is, echo=echo, tech_noise=tech_noise)
            est_results['time'] = t
            return est_results


def process_file(fname):
    with open(fname + '.pickle', 'rb') as f:
        data = pickle.load(f)

    result = data['result']
    processed = dict(
        times=result['N']['time'].tolist(),
        N1_bs=result['N_bs']['mean'][:,0].tolist(),
        N2_bs=result['N_bs']['mean'][:,1].tolist(),
        N1=result['N']['mean'][:,0].tolist(),
        N2=result['N']['mean'][:,1].tolist(),
        visibility=result['V']['mean'].tolist()
        )

    echo = 'echo' in fname
    tech_noise = 'varied_pulse' in fname

    est_vis = []
    est_phnoises = []
    for i in range(len(result['N']['time'])):

        t = result['N']['time'][i]
        Ns = result['N']['values'][i]
        Is = result['I']['values'][i]

        Ns = numpy.concatenate([Ns] * 20, axis=0)
        Is = numpy.concatenate([Is] * 20)

        est_results = fit_visibility(t, Ns, Is, echo=echo, tech_noise=tech_noise)
        est_vis.append(numpy.abs(est_results['amplitude']))
        est_phnoises.append(est_results['phnoise'])

    processed['est_visibility'] = est_vis
    processed['est_phnoises'] = est_phnoises

    return processed


def plot_test_figures(fname):

    processed = process_file(fname)

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.set_ylim(0, 1)
    s.plot(processed['times'], processed['visibility'])
    s.plot(processed['times'], processed['est_visibility'])
    fig.savefig('test_visibility.pdf')

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.set_ylim(0, 55000)
    s.plot(processed['times'], processed['N1'])
    s.plot(processed['times'], processed['N2'])
    fig.savefig('test_population.pdf')

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(processed['times'], processed['est_phnoises'])
    fig.savefig('test_phnoise.pdf')


def process_phnoise_examples():

    sim_phnoise = [
        ('ramsey_wigner', 0.02, 'ramsey_sim_phnoise_20ms.json'),
        ('ramsey_wigner', 0.45, 'ramsey_sim_phnoise_450ms.json')]

    for f_pickle, t, f_json in sim_phnoise:
        with open('processed/' + f_json, 'w') as f:
            json.dump(get_pz(f_pickle, t), f, indent=4)


def process_visibility():

    fnames = [
        'ramsey_gpe_no_losses',
        'ramsey_gpe',
        'echo_gpe',

        'ramsey_wigner',
        'echo_wigner',
        'ramsey_wigner_varied_pulse',
        'echo_wigner_varied_pulse'

        'ramsey_gpe_long',
        'echo_gpe_long',
        'ramsey_wigner_long',
        'echo_wigner_long',
        'ramsey_wigner_varied_pulse_long',
        'echo_wigner_varied_pulse_long',

        'echo_wigner_single_run',
    ]

    for fname in fnames:
        processed = process_file(fname)
        with open('processed/' + fname + '.json', 'w') as f:
            json.dump(processed, f, indent=4)


if __name__ == '__main__':

    process_phnoise_examples()
    process_visibility()
    #plot_test_figures('ramsey_wigner_varied_pulse')
