import numpy
import pickle
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from split_potentials import grid, cutoff


def plot_density(results):
    density = results['axial_density']

    extent=(
        density['time'][0] * 1e3, density['time'][-1] * 1e3,
        -grid.box[-1] / 2 * 1e6, grid.box[-1] / 2 * 1e6)

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.imshow(density['mean'][:,0,:].T,
        vmin=0,
        extent=extent,
        aspect='auto')
    fig.savefig('axial_density_0.pdf')

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.imshow(density['mean'][:,1,:].T,
        vmin=0,
        extent=extent,
        aspect='auto')
    fig.savefig('axial_density_1.pdf')


def plot_position(results):

    density = results['axial_density']

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.set_ylim(-2, 2)
    s.plot(
        density['time'] * 1e3,
        1e6 * (density['mean'][:,0] * grid.xs[-1]).mean(-1) / density['mean'][:,0].mean(-1),
        'b-')
    s.plot(
        density['time'] * 1e3,
        1e6 * (density['mean'][:,1] * grid.xs[-1]).mean(-1) / density['mean'][:,1].mean(-1),
        'r--')
    fig.savefig('position.pdf')


def plot_visibility(results):
    V = results['V']
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.set_ylim(0, 1)
    s.plot(V['time'] * 1e3, V['mean'])
    fig.savefig('visibility.pdf')


def plot_population(results):
    N_bs = results['N_bs']
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.set_ylim(0, 1400)
    s.plot(N_bs['time'] * 1e3, N_bs['mean'][:,0], 'b-')
    s.plot(N_bs['time'] * 1e3, N_bs['mean'][:,1], 'r--')
    s.plot(N_bs['time'] * 1e3, N_bs['mean'].sum(-1), 'g:')
    fig.savefig('population.pdf')


def calculate_squeezing(angles_radian, n, sy, sz):
    # Calculate \Delta^2 \hat{S}_\theta.
    # n, sy, sz and the result have shape (subsets, points)
    subsets, subset_size = n.shape
    tp = angles_radian.size

    tile = lambda arr: numpy.tile(arr.reshape(1, subsets), (tp, 1))

    modes = cutoff.get_modes_number(grid)

    exp_n = tile(n.mean(-1))
    exp_sy = tile(sy.mean(-1))
    exp_sy2 = tile((sy ** 2 - modes / 8.).mean(-1))
    exp_sz = tile(sz.mean(-1))
    exp_sz2 = tile((sz ** 2 - modes / 8.).mean(-1))
    exp_sysz = tile((sy * sz).mean(-1))

    ca = numpy.tile(numpy.cos(angles_radian).reshape(tp, 1), (1, subsets))
    sa = numpy.tile(numpy.sin(angles_radian).reshape(tp, 1), (1, subsets))

    delta_s_theta = (
        ca**2 * (exp_sz2 - exp_sz**2)
        + sa**2 * (exp_sy2 - exp_sy**2)
        - 2 * ca * sa * (exp_sysz - exp_sy * exp_sz))
    return delta_s_theta / exp_n * 4


def get_spins(results):

    I = results['I']['values'][-1]
    Ns = results['N']['values'][-1]
    return Ns.sum(-1), I.real, I.imag, (Ns[:,0] - Ns[:,1]) / 2


def get_tomography(results):

    N, Sx, Sy, Sz = get_spins(results)

    angles_degree = numpy.linspace(-90, 90, 300)
    angles_radian = angles_degree / 180 * numpy.pi

    trj = Sy.size

    subsets = trj // 64
    subset_size = 64

    res_subsets = calculate_squeezing(
        angles_radian,
        N.reshape(subsets, subset_size),
        Sy.reshape(subsets, subset_size),
        Sz.reshape(subsets, subset_size))

    return angles_degree, res_subsets


def plot_tomography(results):

    angles_degree, res_subsets = get_tomography(results)

    fig = plt.figure()
    subplot = fig.add_subplot(111)

    res = res_subsets.mean(1)
    res_err = res_subsets.std(1) // numpy.sqrt(res_subsets.shape[1])
    subplot.plot(angles_degree, numpy.log10(res) * 10)
    #subplot.plot(angles_degree, numpy.log10(res + res_err) * 10, 'b--')
    #subplot.plot(angles_degree, numpy.log10(res - res_err) * 10, 'b--')

    subplot.set_xlim(angles_degree[0], angles_degree[-1])
    subplot.set_ylim(-13, 20)

    fig.savefig('tomography.pdf')


def save_tomography(results, fname):

    angles_degree, res_subsets = get_tomography(results)

    with open(fname, 'w') as f:
        json.dump(dict(
            angles=angles_degree.tolist(),
            squeezing=res_subsets.mean(1).tolist()), f, indent=4)


def save_spins(results, fname):

    _, Sx, Sy, Sz = get_spins(results)

    with open(fname, 'w') as f:
        json.dump(dict(Sx=Sx.tolist(), Sy=Sy.tolist(), Sz=Sz.tolist()), f, indent=4)


if __name__ == '__main__':

    with open('split_potentials.pickle', 'rb') as f:
        data = pickle.load(f)

    print(data['errors'])

    results = data['results']

    #plot_density(results)
    #plot_position(results)
    #plot_visibility(results)
    #plot_population(results)
    plot_tomography(results)
    save_tomography(results, 'riedel_rotation.json')
    save_spins(results, 'riedel_spins.json')

