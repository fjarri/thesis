import numpy
import pickle
import json


def getXiSquared(i, n1, n2):
    """Get squeezing coefficient; see Yun Li et al, Eur. Phys. J. B 68, 365-381 (2009)"""

    # TODO: some generalization required for >2 components

    Si = [i.real, i.imag, 0.5 * (n1 - n2)] # S values for each trajectory

    mSi = numpy.array([x.mean() for x in Si])
    mSij = numpy.array([[
        (Si[i] * Si[j] - (4096 / 8 if i == j else 0)
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
    for is_, n1s, n2s in zip(Is, N1s, N2s):
        xi2s.append(getXiSquared(is_, n1s, n2s))
    return numpy.array(xi2s)


def process_squeezing(losses):
    squeezing = {}
    for a12 in (80., 85., 90., 95.):
        fname = 'feshbach_a12_' + str(a12) + ("" if losses else "_no_losses") + '.pickle'
        with open(fname, 'rb') as f:
            results = pickle.load(f)
        squeezing['times'] = list(results['times'])
        print "Ensembles:", results['N1s'].shape[1]
        squeezing['xi2_' + str(a12)] = list(
            process_squeezing_a12(results['Is'], results['N1s'], results['N2s']))
    return squeezing

with open('feshbach_squeezing.json', 'wb') as f:
    json.dump(process_squeezing(True), f)

with open('feshbach_squeezing_no_losses.json', 'wb') as f:
    json.dump(process_squeezing(False), f)
