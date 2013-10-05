import numpy
import json, pickle
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from figures import get_path
import figures.mplhelpers as mplh
from figures.exact.exact import get_exact_S


def _squeezing(fname, coupling, ens):

    with open(get_path(__file__, 'single_well_squeezing_wigner_' + ens + '_Na200.pickle')) as f:
        wigner = pickle.load(f)

    suffix = '_c' if coupling else '_nc'

    interaction = (100.4, 80.8, 95.5) if coupling else (100.4, 0., 100.4)
    tau_exact = numpy.linspace(1e-3, 120 if coupling else 20, 200)
    s_exact = get_exact_S(tau_exact, 200, *interaction)

    tau_wigner = wigner['tau' + suffix]
    s_wigner = wigner['s_pi2' + suffix]
    s_wigner_err = wigner['s_pi2' + suffix + '_err']


    tau_wigner, s_wigner_bot, s_wigner_top = mplh.crop_bounds(
        tau_wigner, s_wigner - s_wigner_err, s_wigner + s_wigner_err,
        (0, (120 if coupling else 20), 0, 1.2))

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\tau$',
        ylabel='$S_{\\theta + \\pi/2}$')

    s.plot(tau_exact, s_exact, color="black")
    s.fill_between(tau_wigner,
        s_wigner_bot,
        s_wigner_top,
        facecolor=mplh.color.f.blue.lightest,
        linewidth=0,
        #linewidth=0.3,
        #color=mplh.color.f.blue.darkest
        )

    s.text(
        18 if coupling else 3,
        0.9,
        "interaction on" if coupling else "interaction off")

    s.set_xlim((0, 120 if coupling else 20))
    s.set_ylim((0, 1.2))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.text(0.01, 0.92, '(b)' if coupling else '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def _squeezing_err(fname, coupling):

    suffix = '_c' if coupling else '_nc'

    colors = {
        '10k': mplh.color.f.blue,
        '1k': mplh.color.f.red,
        '100': mplh.color.f.green}
    dashes = {
        '10k': '-',
        '1k': '--'}

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\tau$',
        ylabel='Errors for $S_{\\theta + \\pi/2}$')

    for tr in ('1k', '10k'):

        with open(get_path(__file__, 'single_well_squeezing_wigner_' + tr + '_Na200.pickle')) as f:
            wigner = pickle.load(f)

        tau_wigner = wigner['tau' + suffix]
        s_wigner = wigner['s_pi2' + suffix]
        s_wigner_err = wigner['s_pi2' + suffix + '_err']

        interaction = (100.4, 80.8, 95.5) if coupling else (100.4, 0., 100.4)

        s_exact = get_exact_S(tau_wigner, 200, *interaction)
        s_exact[0] = 1.

        diff = numpy.abs(s_wigner - s_exact) / s_exact
        err = s_wigner_err / s_exact

        tau_diff, min_diff, max_diff = mplh.crop_bounds(
            tau_wigner, diff-err, diff+err, (0, (120 if coupling else 20), 0, 0.1))

        s.fill_between(tau_diff, min_diff, max_diff, facecolor=colors[tr].lightest,
            linewidth=0,
            alpha=0.5)
        s.plot(tau_wigner, diff, color=colors[tr].main, dashes=mplh.dash[dashes[tr]])

    s.text(
        72 if coupling else 12,
        0.085,
        "interaction on" if coupling else "interaction off")

    s.set_xlim((0, 120 if coupling else 20))
    s.set_ylim((0, 0.1))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.text(0.01, 0.92, '(b)' if coupling else '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def _squeezing_N_err(fname, coupling):

    suffix = '_c' if coupling else '_nc'

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\tau$',
        ylabel='Errors for $S_{\\theta + \\pi/2}$')

    colors = {
        20: mplh.color.f.blue,
        200: mplh.color.f.red,
        2000: mplh.color.f.green}
    dashes = {
        20: '-',
        200: '--',
        2000: ':'}

    for Na in (20, 200, 2000):

        with open(get_path(__file__, 'single_well_squeezing_wigner_1k_Na' + str(Na) + '.pickle')) as f:
            wigner = pickle.load(f)

        tau_wigner = wigner['tau' + suffix]
        s_wigner = wigner['s_pi2' + suffix]
        s_wigner_err = wigner['s_pi2' + suffix + '_err']

        interaction = (100.4, 80.8, 95.5) if coupling else (100.4, 0., 100.4)

        s_exact = get_exact_S(tau_wigner, Na, *interaction)
        s_exact[0] = 1.

        tau = tau_wigner / tau_wigner[-1]
        diff = numpy.abs(s_wigner - s_exact) / s_exact
        err = s_wigner_err / s_exact

        tau_diff, min_diff, max_diff = mplh.crop_bounds(tau, diff-err, diff+err, (0, 1., 0, 0.2))

        s.fill_between(tau_diff, min_diff, max_diff,
            facecolor=colors[Na].lightest,
            linewidth=0,
            alpha=0.5)
        s.plot(tau, diff, color=colors[Na].main, dashes=mplh.dash[dashes[Na]])

    s.text(
        0.6,
        0.17,
        "interaction on" if coupling else "interaction off")

    s.set_xlim((0, 1.))
    s.set_ylim((0, 0.2))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.text(0.01, 0.92, '(b)' if coupling else '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)



def squeezing_nocc_100(fname):
    _squeezing(fname, False, '100')


def squeezing_cc_100(fname):
    _squeezing(fname, True, '100')


def squeezing_nocc_err(fname):
    _squeezing_err(fname, False)


def squeezing_cc_err(fname):
    _squeezing_err(fname, True)


def squeezing_nocc_N_err(fname):
    _squeezing_N_err(fname, False)


def squeezing_cc_N_err(fname):
    _squeezing_N_err(fname, True)
