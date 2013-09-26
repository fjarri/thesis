import numpy
import json, pickle
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.misc import factorial

from figures import get_path
import figures.mplhelpers as mplh


def G_analytic(gamma, I, J, N):
    if gamma is None:
        s = 0
        for n in xrange(N - J + 1):
            s += factorial(N - n) / factorial(N - J - n)
        return factorial(N) / ((N + 1) * (factorial(N - I))) * s
    else:
        def binom(n, k):
            if k < 0 or k > n: return 0
            return factorial(n) / factorial(k) / factorial(n - k)
        s = 0
        for n in xrange(N - J + 1):
            for i in xrange(I + 1):
                s += factorial(n) * factorial(N - n) ** 2 / factorial(N - I) / factorial(N - J - n) * \
                    binom(I, i) * binom(N - J, n - i) * (1 - gamma) ** i * gamma ** (I - i)
        return s / (N + 1)

def g_analytic(theta, J, N):
    gamma = numpy.cos(theta) ** 2
    return G_analytic(gamma, J, J, N) / G_analytic(None, J, J, N)

def deltas_analytic(thetas, J, N):
    gs = g_analytic(thetas, J, N)
    gs_3theta = g_analytic(thetas * 3, J, N)
    return (3 * gs - gs_3theta - 2)


def _cooperative(fname, N):
    if N == 1:
        data = 'cooperative-N1-J1-21.json'
    else:
        data = 'cooperative-N2-J2-25.json'

    with open(get_path(__file__, data)) as f:
        n = json.load(f)

    fig = mplh.figure(width=0.5)
    ax = fig.add_subplot(111)

    thetas_scaled = numpy.array(n['thetas'])
    deltas = numpy.array(n['deltas_mean'])
    err = numpy.array(n['deltas_err'])

    ax.fill_between(thetas_scaled, deltas-err, deltas+err,
        facecolor=mplh.color.f.blue.lightest, interpolate=True,
        color=mplh.color.f.blue.darkest,
        linewidth=0.3)

    ax.plot(
        thetas_scaled,
        deltas_analytic(thetas_scaled / numpy.sqrt(N), N, N),
            'k--', dashes=mplh.dash['--'])

    ax.set_xlim((thetas_scaled[0], thetas_scaled[-1]))
    ax.set_ylim((-0.05, 0.45 if N == 1 else 0.55))

    ax.set_xlabel("$\\theta" + ("\\sqrt{2}" if N == 2 else "") + "$ (rad)")
    ax.set_ylabel("$\\Delta$")

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def cooperative_N1(fname):
    _cooperative(fname, 1)


def cooperative_N2(fname):
    _cooperative(fname, 2)
