import numpy
import json, pickle
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from figures import get_path
import figures.mplhelpers as mplh


def _squeezing(fname, coupling):
    with open(get_path(__file__, 'single_well_squeezing_exact.pickle')) as f:
        exact = pickle.load(f)
    with open(get_path(__file__, 'single_well_squeezing_wigner.pickle')) as f:
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

    s.set_xlim((0, 120 if coupling else 20))
    s.set_ylim((0, 1.2))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)



def squeezing_nocc(fname):
    _squeezing(fname, False)


def squeezing_cc(fname):
    _squeezing(fname, True)
