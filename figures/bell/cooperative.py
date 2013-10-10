import json, os, os.path

import numpy
from scipy.misc import factorial
from scipy.special import gamma as gamma_func
from numpy.random import gamma as gamma_sample
from numpy.random import normal, rand

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Target distribution for \alpha^\prime and \beta^\prime
def Pprime(alpha_p, beta_p, N):
    alpha_abs = (numpy.abs(alpha_p) ** 2).sum(0)
    beta_abs = (numpy.abs(beta_p) ** 2).sum(0)
    return numpy.abs(
            alpha_p[0] * beta_p[0] + alpha_p[1] * beta_p[1]
        ) ** (2 * N) / \
        (numpy.pi ** 4 * (N + 1) * factorial(N) ** 2) * \
        numpy.exp(-alpha_abs - beta_abs)

# Bounding distribution for \alpha^\prime and \beta^\prime
def Ptilde(alpha_p, beta_p, N):
    alpha_abs = (numpy.abs(alpha_p) ** 2).sum(0)
    beta_abs = (numpy.abs(beta_p) ** 2).sum(0)

    coeff = (numpy.pi ** 2 * factorial(N + 1)) ** 2

    return (alpha_abs ** N * numpy.exp(-alpha_abs) *
        beta_abs ** N * numpy.exp(-beta_abs) / coeff)

# calculate vectors randomly distributed on k-dim unit sphere
def sphere_uniform(size, dims):
    coords = normal(size=(dims, size))
    rnd_lengths = numpy.tile(numpy.sqrt((coords ** 2).sum(0)), (dims, 1))
    return coords / rnd_lengths


def samplePprime(N, Nrandoms):
    """
    Generates ``Nrandoms`` samples of the bounding distribution and returns only
    those that fall into target distribution
    """

    # sample values of |\vec{\alpha}^\prime|^2 and |\vec{\beta}^\prime|^2
    l_alpha_p = numpy.sqrt(gamma_sample(N + 2, size=Nrandoms))
    l_beta_p = numpy.sqrt(gamma_sample(N + 2, size=Nrandoms))

    alpha_p_real = sphere_uniform(Nrandoms, 4) * l_alpha_p
    beta_p_real = sphere_uniform(Nrandoms, 4) * l_beta_p

    alpha_p = alpha_p_real[:2] + 1j * alpha_p_real[2:]
    beta_p = beta_p_real[:2] + 1j * beta_p_real[2:]

    M_squared = (N + 1)

    u = rand(Nrandoms)
    P = Pprime(alpha_p, beta_p, N)

    bound = M_squared * Ptilde(alpha_p, beta_p, N)
    assert (P <= bound).all()

    sample_nums = (u < P / bound)
    alpha_p = alpha_p.T[sample_nums].T
    beta_p = beta_p.T[sample_nums].T

    return alpha_p, beta_p

def sampleDeterministic(N, size):
    """
    Returns exactly ``size`` of samples.
    """
    alpha_res = []
    beta_res = []
    good_samples = 0
    total_samples = 0

    limit = 10000000
    Nestimate = min(size, limit)

    while 1:
        print "Requesting", Nestimate, "values..."
        alpha_p, beta_p = samplePprime(N, Nestimate)
        total_samples += Nestimate
        good_samples += alpha_p.shape[1]
        print "Got", good_samples, "good samples total (" + \
            str(int(good_samples / float(size) * 100)) + "%)"
        alpha_res.append(alpha_p)
        beta_res.append(beta_p)

        if good_samples >= size:
            break

        if good_samples == 0:
            Nestimate *= 2
        else:
            success_coeff = float(good_samples) / total_samples
            Nestimate = int(1.1 * (size - good_samples) / success_coeff)
        Nestimate = min(Nestimate, limit)

    print "Success coefficient:", float(good_samples) / total_samples
    return numpy.concatenate(alpha_res, axis=1)[:,:size], \
        numpy.concatenate(beta_res, axis=1)[:,:size]

def sampleP(N, Nsamples):
    alpha_prime, beta_prime = sampleDeterministic(N, Nsamples)
    normals = numpy.random.normal(scale=numpy.sqrt(0.5), size=(8, Nsamples))
    dalpha = normals[0:2] + 1j * normals[2:4]
    dbeta = normals[4:6] + 1j * normals[6:8]
    alpha = numpy.concatenate([alpha_prime + dalpha, (alpha_prime - dalpha).conj()], axis=0)
    beta = numpy.concatenate([beta_prime + dbeta, (beta_prime - dbeta).conj()], axis=0)
    return alpha, beta

def A(gamma, J, vec):
    """
    ``vec`` must have shape (4, subsets, samples)
    Returns: (subsets, samples)
    """
    if gamma is not None:
        return (gamma ** 0.5 * vec[0] + (1 - gamma) ** 0.5 * vec[1]) ** J * \
            (gamma ** 0.5 * vec[2] + (1 - gamma) ** 0.5 * vec[3]) ** J
    else:
        return (vec[0] * vec[2] + vec[1] * vec[3]) ** J

def G(gamma, I, J, alpha, beta):
    """
    ``alpha`` and ``beta`` must have shape (4, subsets, samples)
    Returns (subsets,)
    """
    return (A(1, I, alpha) * A(gamma, J, beta)).mean(1)

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

def g(theta, J, alpha, beta):
    """
    ``alpha`` and ``beta`` must have shape (4, subsets, samples)
    """
    gamma = numpy.cos(theta) ** 2
    return G(gamma, J, J, alpha, beta) / G(None, J, J, alpha, beta)

def g_analytic(theta, J, N):
    gamma = numpy.cos(theta) ** 2
    return G_analytic(gamma, J, J, N) / G_analytic(None, J, J, N)

def deltas(thetas, J, alpha, beta):
    """
    ``alpha`` and ``beta`` must have shape (4, subsets, samples)
    """
    subsets = alpha.shape[1]
    gs = numpy.concatenate(
        [g(theta, J, alpha, beta).reshape(subsets, 1) for theta in thetas], axis=1)
    gs_3theta = numpy.concatenate(
        [g(3 * theta, J, alpha, beta).reshape(subsets, 1) for theta in thetas], axis=1)
    # taking the real part, because the imaginary part only contains garbage from imperfect sampling
    return (3 * gs - gs_3theta - 2).real

def deltas_analytic(thetas, J, N):
    gs = g_analytic(thetas, J, N)
    gs_3theta = g_analytic(thetas * 3, J, N)
    return (3 * gs - gs_3theta - 2)

def getDeltas(N, J, Nsamples):
    thetas = numpy.linspace(0, 0.5 / numpy.sqrt(J))
    alpha, beta = sampleP(N, Nsamples)

    # estimate errors
    # take number of subsets close to sqrt(Nsamples); assumes Nsamples == 2 ** n
    subsets = 1024 # 2 ** (int(numpy.log2(Nsamples)) / 2)
    assert Nsamples % subsets == 0

    alpha = alpha.reshape(4, 1, Nsamples)
    beta = beta.reshape(4, 1, Nsamples)

    print (alpha[0] * alpha[2] + alpha[1] * alpha[3]).mean()
    print (beta[0] * beta[2] + beta[1] * beta[3]).mean()

    alpha_sb = alpha.reshape(4, subsets, Nsamples / subsets)
    beta_sb = beta.reshape(4, subsets, Nsamples / subsets)

    deltas_mean = deltas(thetas, J, alpha, beta)[0]
    deltas_sb = deltas(thetas, J, alpha_sb, beta_sb)
    errors = deltas_sb.std(0) / numpy.sqrt(subsets)

    deltas_a = deltas_analytic(thetas, J, N)

    thetas_scaled = thetas * numpy.sqrt(J)

    return thetas_scaled, deltas_a, deltas_mean, errors

def plotDeltas(thetas, deltas_a, deltas_mean, deltas_err):
    fig = matplotlib.pyplot.figure()
    s = fig.add_subplot(111, xlabel="$\\theta / \\sqrt{J}$", ylabel="$\\Delta(\\theta)$")
    s.errorbar(thetas, deltas_mean, yerr=deltas_err, color='b')
    s.plot(thetas, deltas_a, 'r--')
    s.set_xlim(xmin=thetas[0], xmax=thetas[-1])
    s.set_ylim(ymin=deltas_a.min() - 0.05, ymax=deltas_a.max() + 0.05)
    return fig


def sample(N, J, Nsamples):
    name = 'cooperative-N{N}-J{J}-{samples}'.format(N=N, J=J, samples=int(numpy.log2(Nsamples)))
    json_name = name + '.json'

    if not os.path.exists(json_name):
        thetas, deltas_a, deltas_mean, deltas_err = getDeltas(N, J, Nsamples)
        json.dump(dict(
                thetas=thetas.tolist(),
                deltas_mean=deltas_mean.tolist(),
                deltas_err=deltas_err.tolist(),
                deltas_a=deltas_a.tolist()
            ),
            open(json_name, 'w'))


    data = json.load(open(json_name))
    thetas = numpy.array(data['thetas'])
    deltas_a = numpy.array(data['deltas_a'])
    deltas_mean = numpy.array(data['deltas_mean'])
    deltas_err = numpy.array(data['deltas_err'])

    plotDeltas(thetas, deltas_a, deltas_mean, deltas_err).savefig(name + '.pdf')



for n in (17, 18, 19, 20, 21):
   sample(1, 1, 2 ** n)

for n in (17, 18, 19, 20, 21):
    sample(2, 1, 2 ** n)

for n in (19, 21, 23, 25):
    sample(2, 2, 2 ** n)
