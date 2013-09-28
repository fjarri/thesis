import numpy
import json, pickle
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.misc import factorial

from figures import get_path
import figures.mplhelpers as mplh
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib


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

    ax.set_xlabel(
        "$\\phantom{\\sqrt{2}}$" +
        "$\\theta" + ("\\sqrt{2}" if N == 2 else "") + "$ (rad)"
        + "$\\phantom{\\sqrt{2}}$")
    ax.set_ylabel("$\\Delta$")

    ax.text(0.3,
        -0.05 + (0.05 if N == 1 else 0.05 * (0.6 / 0.5)),
        '$N=1$, $J=1$' if N == 1 else '$N=2$, $J=2$')

    fig.text(0.01, 0.92, '(a)' if N == 1 else '(b)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def cooperative_N1(fname):
    _cooperative(fname, 1)


def cooperative_N2(fname):
    _cooperative(fname, 2)


def _distribution(fname, distr_type, order):

    if distr_type == 'Q':
        data_name = 'ghz_binning_ardehali_2p_Q.pickle'
    else:
        data_name = 'ghz_binning_ardehali_2p_number.pickle'


    with open(get_path(__file__, data_name)) as f:
        data = pickle.load(f)

    if order == 1:
        corrs = data[1][0]
        zmax = 5.5
    else:
        corrs = data[1][1]
        zmax = 3.5

    data, edges = corrs

    fig = mplh.figure(width=0.5, aspect=0.7)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=40., azim=245.)

    X = (edges[0][1:] + edges[0][:-1]) / 2
    Y = (edges[1][1:] + edges[1][:-1]) / 2

    # normalize on 1
    data = data.astype(numpy.float64) / data.sum() / (X[1] - X[0]) / (Y[1] - Y[0]) * 100

    X, Y = numpy.meshgrid(X, Y)

    ax.contour(X, Y, data.T, cmap=mplh.cm_zeropos,
        levels=numpy.linspace(0, zmax, 25))

    ax.set_zlabel('\n\nprobability ($\\times 10^{-2}$)')

    if order == 1:
        ax.set_xlabel('\n\n$\\mathrm{Re}\,\\sigma_1^x$')
        ax.set_ylabel('\n\n$\\mathrm{Re}\,\\sigma_2^x$')
        #ax.set_zlabel('\n\n$P_{\\mathrm{' + representation + '}}$, $\\times 10^{-2}$')

        ax.set_xlim3d(-3.5, 3.5)
        ax.xaxis.set_ticks(range(-3, 4))
        ax.yaxis.set_ticks(range(-3, 4))
    else:
        ax.set_xlabel('\n\n$\\mathrm{Re}\,\\sigma_1^x \\sigma_2^x$')
        ax.set_ylabel('\n\n$\\mathrm{Re}\,\\sigma_1^y \\sigma_2^y$')
        #ax.set_zlabel('\n\n$P_{\\mathrm{' + representation + '}}$, $\\times 10^{-2}$')

        ax.set_xlim3d(-8.5, 6.5)
        ax.set_ylim3d(-6.5, 8.5)
        ax.xaxis.set_ticks(range(-8, 8, 2))
        ax.yaxis.set_ticks(range(-6, 10, 2))
        ax.zaxis.set_ticks([0, 1, 2, 3])

    ax.set_zlim3d(0, zmax)

    # clear background panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    #ax.w_xaxis.set_rotate_label(False)
    #ax.w_yaxis.set_rotate_label(False)

    fig.text(0.4, 0.88, 'SU(2)-Q' if distr_type == 'Q' else 'positive-P')

    labels = {
        ('P', 1): '(a)',
        ('P', 2): '(b)',
        ('Q', 1): '(c)',
        ('Q', 2): '(d)',
    }

    fig.text(0.01, 0.92, labels[(distr_type, order)], fontweight='bold')

    fig.tight_layout(pad=1.3)
    fig.savefig(fname)


def distribution_Q1(fname):
    _distribution(fname, 'Q', 1)

def distribution_Q2(fname):
    _distribution(fname, 'Q', 2)

def distribution_P1(fname):
    _distribution(fname, 'P', 1)

def distribution_P2(fname):
    _distribution(fname, 'P', 2)


def getF_analytical(particles, quantity):
    """
    Returns 'classical' and 'quantum' predictions for the
    Mermin's/Ardehali's state and operator.
    """
    if quantity == 'F_mermin':
        return 2. ** (particles / 2), 2. ** (particles - 1)
    elif quantity == 'F_ardehali':
        return 2. ** ((particles + 1) / 2), 2. ** (particles - 0.5)
    else:
        raise NotImplementedError(quantity)


def filter_data(data, **kwds):
    result = []
    for d in data:
        for key in kwds:
            if kwds[key] != d[key]:
                break
        else:
            result.append(d)

    ns = []
    vals = []
    errs = []
    lhvs = []
    qms = []
    for r in sorted(result, key=lambda x: x['particles']):
        if r['quantity'] in ('F_ardehali', 'F_mermin'):
            cl, qm = getF_analytical(r['particles'], r['quantity'])
            if r['error'] / qm > 0.5:
                continue
            lhvs.append(cl)
            qms.append(qm)

        ns.append(r['particles'])
        vals.append(r['mean'])
        errs.append(r['error'])

    return dict(ns=numpy.array(ns), mean=numpy.array(vals), error=numpy.array(errs),
        lhvs=numpy.array(lhvs), qms=numpy.array(qms))


def ghz_violations(fname):
    with open(get_path(__file__, 'ghz_sampling.json')) as f:
        data = json.load(f)

    fig = mplh.figure(width=0.5)

    G = matplotlib.gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(G[0,0])
    ax2 = fig.add_subplot(G[0,1])

    ax1.set_xlabel('$M$',
        color='white'
        ) # need it to make matplotlib create proper spacing
    fig.text(0.55, 0.04, '$M$')
    ax1.set_ylabel('$F / F_{\\mathrm{QM}}$')

    violations = filter_data(data['violations'], representation='Q', size=10**9)
    violations_p = filter_data(data['violations'], representation='number', size=10**9)

    ns = violations['ns']
    qms = violations['qms']
    mean = violations['mean'] / qms
    err = violations['error'] / qms

    ns_p = violations_p['ns']
    qms_p = violations_p['qms']
    mean_p = violations_p['mean'] / qms_p
    err_p = violations_p['error'] / qms_p

    cl_ns = numpy.arange(1, 61)
    cl_qm = [getF_analytical(n, 'F_ardehali' if n % 2 == 0 else 'F_mermin') for n in cl_ns]
    cl_qm = numpy.array(zip(*cl_qm)[0]) / numpy.array(zip(*cl_qm)[1])

    ax1.set_xlim((0, 10.5))
    ax1.set_ylim((-0.05, 1.6))
    ax2.set_xlim((49.5, 61))
    ax2.set_ylim((-0.05, 1.6))

    for ax in (ax1, ax2):
        ax.plot(cl_ns, numpy.ones(60), color='grey', linewidth=0.5,
            linestyle='--', dashes=mplh.dash['--'])
        ax.errorbar(ns, mean, yerr=err, color=mplh.color.f.blue.main, linestyle='None',
            capsize=1.5)
        ax.plot(cl_ns, cl_qm, color=mplh.color.f.yellow.main, linestyle='-.', dashes=mplh.dash['-.'])

    ax1.errorbar(ns_p, mean_p, yerr=err_p, color=mplh.color.f.red.main, linestyle='None',
        capsize=1.5)

    ax1.text(5, 0.37, '\\textsc{lhv}')
    ax2.text(51, 0.5, 'SU(2)-Q')

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax1.xaxis.set_ticks([1, 5, 10])
    ax2.xaxis.set_ticks([50, 55, 60])
    ax2.tick_params(labelright='off') # don't put tick labels at the right side
    ax1.yaxis.tick_left()

    # add cut-out lines
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=0.5)
    ax2.plot((-d,+d),(-d,+d), **kwargs)
    ax2.plot((-d,+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax1.transAxes, linewidth=0.5)  # switch to the bottom axes
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
    ax1.plot((1-d,1+d),(-d,+d), **kwargs)

    #fig.subplots_adjust(wspace=0.001)

    fig.text(0.01, 0.92, '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)

    #p1 = ax1.get_position()
    #p2 = ax2.get_position()

    #dwidth = (p2.x0 - p1.x0 - p1.width) / 2. - 0.01
    #ax1.set_position([p1.x0, p1.y0, p1.width + dwidth, p1.height])
    #ax2.set_position([p2.x0 - dwidth, p2.y0, p2.width + dwidth, p2.height])

    fig.savefig(fname)


def ghz_errors(fname):
    with open(get_path(__file__, 'ghz_sampling.json')) as f:
        data = json.load(f)

    fig = mplh.figure(width=0.5)

    ax = fig.add_subplot(111)
    ax.set_xlabel('$M$')
    ax.set_ylabel('$\\log_{2}(\\mathrm{Err}(F) / F_{\\mathrm{QM}})$')

    corr1 = filter_data(data['different_order_correlations'],
        representation='Q', quantity='N_total', size=10**9)
    corrm = filter_data(data['violations'],
        representation='Q', size=10**9)

    corrp = filter_data(data['violations'],
        representation='number', size=10**9)

    ax.plot(corr1['ns'], numpy.log2(corr1['error'] / corr1['ns'] * 2.),
        color=mplh.color.f.green.main, linestyle='-.', dashes=mplh.dash['-.'])
    ax.plot(corrm['ns'], numpy.log2(corrm['error'] / corrm['qms'])[:50],
        color=mplh.color.f.blue.main)
    ax.plot(corrp['ns'], numpy.log2(corrp['error'] / corrp['qms']),
        color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])

    ref_ns = numpy.arange(1, 36)
    ax.plot(ref_ns, ref_ns / 2. - 20, linestyle=':',
        dashes=mplh.dash[':'], linewidth=0.5, color='grey')

    ax.set_xlim((0, 61))
    ax.set_ylim((-24, 0))
    ax.yaxis.set_ticks(range(-20, 1, 5))

    ax.text(40, -21, 'first order')
    ax.text(40, -10, 'SU(2)-Q')
    ax.text(18, -3, 'positive-P')
    ax.text(34, -5, 'reference')

    fig.text(0.01, 0.92, '(b)', fontweight='bold')

    fig.tight_layout(pad=0.3)

    fig.savefig(fname)
