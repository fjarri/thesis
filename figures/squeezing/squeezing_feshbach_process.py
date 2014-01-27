import numpy
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def _feshbach_squeezing(fname, losses):

    with open(get_path(__file__, 'feshbach_squeezing' + ('' if losses else '_no_losses') + '.json')) as f:
        sq = json.load(f)

    datasets = [
        ('80.0', mplh.color.f.blue, '-', ),
        ('85.0', mplh.color.f.red, '--'),
        ('90.0', mplh.color.f.green, ':'),
        ('95.0', mplh.color.f.yellow, '-.'),
    ]

    t_sq = numpy.array(sq['times'])

    fig = mplh.figure(width=0.5)
    subplot = fig.add_subplot(111)

    for a12, color, linestyle in datasets:
        xi2_sq = numpy.array(sq['xi2_' + a12])
        xi2_sq_err = numpy.array(sq['xi2_' + a12 + '_err'])

        xi2_log = 10 * numpy.log10(xi2_sq)
        err_down = 10 * numpy.log10(xi2_sq - xi2_sq_err)
        err_up = 10 * numpy.log10(xi2_sq + xi2_sq_err)

        t_err, err_down, err_up = mplh.crop_bounds(
            t_sq, err_down, err_up, (0, 0.1, (-13 if losses else -20), 1))

        positive = err_up > err_down
        subplot.fill_between(
            t_err * 1e3, err_down, err_up,
            facecolor=color.lightest,
            #interpolate=True,
            linewidth=0)

        subplot.plot(t_sq * 1e3, xi2_log, color=color.main,
            linestyle=linestyle, dashes=mplh.dash[linestyle])

    subplot.plot([0, 100], [0, 0], color='grey', linewidth=0.5,
        linestyle='-.', dashes=mplh.dash['-.'])

    subplot.set_xlim(xmin=0, xmax=100)
    subplot.set_ylim(ymin=-13 if losses else -20 , ymax=1)
    subplot.set_xlabel('$T$ (ms)')
    subplot.set_ylabel('$\\xi^2$ (dB)')

    subplot.text(75,
        1 - 2.5 if losses else 1 - (2.5 / 14. * 21.),
        '1-2 losses' if losses else 'no losses')

    if losses:
        subplot.text(42, -3, '$80.0\,r_B$')
        subplot.text(75, -11, '$85.0\,r_B$')
        subplot.text(20, -9, '$90.0\,r_B$')
        subplot.text(43, -7.5, '$95.0\,r_B$')
    else:
        subplot.text(9, -18, '$80.0\,r_B$')
        subplot.text(41.5, -18, '$85.0\,r_B$')
        subplot.text(60, -15, '$90.0\,r_B$')
        subplot.text(70, -9.5, '$95.0\,r_B$')

    fig.text(0.01, 0.92, '(b)' if losses else '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def getXiSquared(i, n1, n2):
    """Get squeezing coefficient; see Yun Li et al, Eur. Phys. J. B 68, 365-381 (2009)"""

    # TODO: some generalization required for >2 components

    Si = [i.real, i.imag, 0.5 * (n1 - n2)] # S values for each trajectory

    mSi = numpy.array([x.mean() for x in Si])
    mSij = numpy.array([[
        (0.5 * (Si[i] * Si[j] + Si[j] * Si[i])
            - (4096 / 8 if i == j else 0)
            ).mean()
        for j in (0,1,2)] for i in (0,1,2)])

    mSi2 = numpy.dot(mSi.reshape(3,1), mSi.reshape(1,3))
    deltas = mSij - mSi2

    S = numpy.sqrt(mSi[0] ** 2 + mSi[1] ** 2 + mSi[2] ** 2) # <S>

    phi = numpy.arctan2(mSi[1], mSi[0]) # azimuthal angle of S
    yps = numpy.arccos(mSi[2] / S) # polar angle of S

    sin = numpy.sin
    cos = numpy.cos

    A = (sin(phi) ** 2 - cos(yps) ** 2 * cos(phi) ** 2) * deltas[0, 0] + \
        (cos(phi) ** 2 - cos(yps) ** 2 * sin(phi) ** 2) * deltas[1, 1] - \
        sin(yps) ** 2 * deltas[2, 2] - \
        (1 + cos(yps) ** 2) * sin(2 * phi) * deltas[0, 1] + \
        sin(2 * yps) * cos(phi) * deltas[2, 0] + \
        sin(2 * yps) * sin(phi) * deltas[1, 2]

    B = cos(yps) * sin(2 * phi) * (deltas[0, 0] - deltas[1, 1]) - \
        2 * cos(yps) * cos(2 * phi) * deltas[0, 1] - \
        2 * sin(yps) * sin(phi) * deltas[2, 0] + \
        2 * sin(yps) * cos(phi) * deltas[1, 2]

    Sperp_squared = \
        0.5 * (cos(yps) ** 2 * cos(phi) ** 2 + sin(phi) ** 2) * deltas[0, 0] + \
        0.5 * (cos(yps) ** 2 * sin(phi) ** 2 + cos(phi) ** 2) * deltas[1, 1] + \
        0.5 * sin(yps) ** 2 * deltas[2, 2] - \
        0.5 * sin(yps) ** 2 * sin(2 * phi) * deltas[0, 1] - \
        0.5 * sin(2 * yps) * cos(phi) * deltas[2, 0] - \
        0.5 * sin(2 * yps) * sin(phi) * deltas[1, 2] - \
        0.5 * numpy.sqrt(A ** 2 + B ** 2)

    Na = n1.mean()
    Nb = n2.mean()

    return (Na + Nb) * Sperp_squared / (S ** 2)


def process_squeezing_a12(Is, N1s, N2s):
    xi2s = []
    xi2s_err = []

    for is_, n1s, n2s in zip(Is, N1s, N2s):
        xi2s.append(getXiSquared(is_, n1s, n2s))

        full_size = is_.size
        chunk_num = 16
        chunk_size = full_size / chunk_num

        sub_xi2 = []
        for i in range(chunk_num):
            i_start = i * chunk_size
            i_end = (i + 1) * chunk_size
            sub_xi2.append(getXiSquared(is_[i_start:i_end], n1s[i_start:i_end], n2s[i_start:i_end]))

        xi2s_err.append(numpy.array(sub_xi2).std() / numpy.sqrt(chunk_num))

    return numpy.array(xi2s), numpy.array(xi2s_err)


def process_squeezing(losses):
    squeezing = {}
    for a12 in (80., 85., 90., 95.):
        fname = 'feshbach_a12_' + str(a12) + ("" if losses else "_no_losses") + '.pickle'
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        results = data['results']
        Is = results['I']['values']
        Ns = results['N']['values']
        N1s = Ns[:,:,0]
        N2s = Ns[:,:,1]
        times = results['I']['time']

        print(losses, a12, Ns.mean(1).sum(1)[0], Ns.mean(1).sum(1)[-1])
        print(data['a12'], data['losses_enabled'], data['errors'])

        xi2s, xi2s_err = process_squeezing_a12(Is, N1s, N2s)
        squeezing['times'] = times.tolist()
        squeezing['xi2_' + str(a12)] = xi2s.tolist()
        squeezing['xi2_' + str(a12) + '_err'] = xi2s_err.tolist()

        print(xi2s[0], xi2s[-1])

    return squeezing


with open('feshbach_squeezing.json', 'wb') as f:
    json.dump(process_squeezing(True), f)


with open('feshbach_squeezing_no_losses.json', 'wb') as f:
    json.dump(process_squeezing(False), f)
