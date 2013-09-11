import numpy
import pickle

from figures import get_path
import figures.mplhelpers as mplh


def _one_comp_gs(fname, N):
    with open(get_path(__file__, 'one_comp_gs.pickle'), 'rb') as f:
        data = pickle.load(f)

    zs = data[N]['zs'] / 1e-6 # to um
    n_z = data[N]['n_z'] * 1e-6 # to um^-1
    tf_n_z = data[N]['tf_n_z'] * 1e-6 # to um^-1

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$z$ ($\\mu\\mathrm{m}$)',
        ylabel='Axial density ($\\mu\\mathrm{m}^{-1}$)')
    s.plot(zs, n_z[0], color=mplh.color.f.blue.main, linestyle='-')
    s.plot(zs, tf_n_z[0], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def one_comp_gs_small(fname):
    _one_comp_gs(fname, 1000)


def one_comp_gs_large(fname):
    _one_comp_gs(fname, 100000)


def _two_comp_gs(fname, a12):
    with open(get_path(__file__, 'two_comp_gs.pickle'), 'rb') as f:
        data = pickle.load(f)

    zs = data[a12]['zs'] / 1e-6 # to um
    n_z = data[a12]['n_z'] * 1e-6 # to um^-1

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$z$ ($\\mu\\mathrm{m}$)',
        ylabel='Axial density ($\\mu\\mathrm{m}^{-1}$)')
    s.plot(zs, n_z[0], color=mplh.color.f.blue.main, linestyle='-')
    s.plot(zs, n_z[1], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def two_comp_gs_miscible(fname):
    _two_comp_gs(fname, 97.0)


def two_comp_gs_immiscible(fname):
    _two_comp_gs(fname, 99.0)
