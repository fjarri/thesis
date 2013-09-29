import numpy
import json, pickle
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from figures import get_path
import figures.mplhelpers as mplh


def _squeezing(fname, coupling, ens):
    with open(get_path(__file__, 'single_well_squeezing_exact.pickle')) as f:
        exact = pickle.load(f)
    with open(get_path(__file__, 'single_well_squeezing_wigner_' + ens + '.pickle')) as f:
        wigner = pickle.load(f)

    suffix = '_c' if coupling else '_nc'

    tau_exact = exact['tau' + suffix]
    s_exact = exact['s_pi2' + suffix]
    tau_wigner = wigner['tau' + suffix]
    s_wigner = wigner['s_pi2' + suffix]
    s_wigner_err = wigner['s_pi2' + suffix + '_err']

    s_wigner_bot = s_wigner - s_wigner_err
    s_wigner_top = s_wigner + s_wigner_err
    drop = None
    for i in range(s_wigner_top.size):
        if s_wigner_top[i] > 1.2:
            drop = i
            break
        if tau_wigner[i] > (120 if coupling else 20):
            drop = i
            break
    else:
        drop = s_wigner_top.size - 1

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\tau$',
        ylabel='$S^{\\theta + \\pi/2}$')

    s.plot(tau_exact, s_exact, color="black")
    s.fill_between(tau_wigner[:drop],
        s_wigner_bot[:drop],
        s_wigner_top[:drop],
        facecolor=mplh.color.f.blue.lightest,
        linewidth=0)

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
    with open(get_path(__file__, 'single_well_squeezing_exact.pickle')) as f:
        exact = pickle.load(f)
    with open(get_path(__file__, 'single_well_squeezing_wigner_100.pickle')) as f:
        wigner_100 = pickle.load(f)
    with open(get_path(__file__, 'single_well_squeezing_wigner_1k.pickle')) as f:
        wigner_1k = pickle.load(f)
    with open(get_path(__file__, 'single_well_squeezing_wigner_10k.pickle')) as f:
        wigner_10k = pickle.load(f)


    suffix = '_c' if coupling else '_nc'

    tau_exact = exact['tau' + suffix]
    s_exact = exact['s_pi2' + suffix]
    if not coupling:
        tau_exact = tau_exact[:200]
        s_exact = s_exact[:200]

    tau_wigner_100 = wigner_100['tau' + suffix]
    s_wigner_100 = wigner_100['s_pi2' + suffix]
    s_wigner_err_100 = wigner_100['s_pi2' + suffix + '_err']

    tau_wigner_1k = wigner_1k['tau' + suffix]
    s_wigner_1k = wigner_1k['s_pi2' + suffix]
    s_wigner_err_1k = wigner_1k['s_pi2' + suffix + '_err']

    tau_wigner_10k = wigner_10k['tau' + suffix]
    s_wigner_10k = wigner_10k['s_pi2' + suffix]
    s_wigner_err_10k = wigner_10k['s_pi2' + suffix + '_err']

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\tau$',
        ylabel='Errors for $S^{\\theta + \\pi/2}$')

    exact_interp = interp1d(tau_exact, s_exact, kind="cubic", bounds_error=False)

    s.plot(tau_wigner_100,
        numpy.abs(exact_interp(tau_wigner_100) - s_wigner_100) / exact_interp(tau_wigner_100),
        color=mplh.color.f.blue.main)
    s.plot(tau_wigner_100,
        s_wigner_err_100 / exact_interp(tau_wigner_100),
        color=mplh.color.f.blue.main, linestyle='--', dashes=mplh.dash['--'])
    s.plot(tau_wigner_1k,
        numpy.abs(exact_interp(tau_wigner_1k) - s_wigner_1k) / exact_interp(tau_wigner_1k),
        color=mplh.color.f.red.main)
    s.plot(tau_wigner_1k,
        s_wigner_err_1k / exact_interp(tau_wigner_1k),
        color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])

    """
    s.plot(tau_wigner_10k,
        numpy.abs(exact_interp(tau_wigner_10k) - s_wigner_10k) / exact_interp(tau_wigner_10k),
        color=mplh.color.f.green.main)
    s.plot(tau_wigner_10k,
        s_wigner_err_10k / exact_interp(tau_wigner_10k),
        color=mplh.color.f.green.main, linestyle='--', dashes=mplh.dash['--'])
    """

    s.text(
        18 if coupling else 3,
        0.13,
        "interaction on" if coupling else "interaction off")

    s.set_xlim((0, 120 if coupling else 20))
    s.set_ylim((0, 0.15))

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
