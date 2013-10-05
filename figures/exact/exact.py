import numpy


# Exact calculations for the spin squeezing,
# courtesy of Run Yan Teh
def get_exact_S(times, N, a11, a12, a22):

    alpha = numpy.sqrt(N)

    g11 = 1. / N
    g12 = a12 / a11 / N
    g21 = a12 / a11 / N
    g22 = a22 / a11 / N

    ca1_a2 = (alpha ** 2 / 2
        * numpy.exp(alpha ** 2 / 2 * (numpy.exp(1j*(g11-g21)*times) - 1))
        * numpy.exp(alpha ** 2 / 2 * (numpy.exp(1j*(g12-g22)*times) - 1)))
    ca2_a1 = ca1_a2.conj()
    ca1_a1 = alpha ** 2 / 2
    ca2_a2 = alpha ** 2 / 2

    ca1_a2_ca1_a2 = (alpha ** 4 / 4
        * numpy.exp(alpha ** 2 / 2 * (numpy.exp(2j*(g11-g21)*times) - 1))
        * numpy.exp(alpha ** 2 / 2 * (numpy.exp(2j*(g12-g22)*times) - 1)))
    ca2_a1_ca2_a1 = ca1_a2_ca1_a2.conj()

    ca1_a2_ca2_a2 = ca1_a2 + alpha ** 2 / 2 * ca1_a2 * numpy.exp(1j*(g12-g22)*times)
    ca2_a1_ca1_a1 = ca2_a1 + alpha ** 2 / 2 * ca2_a1 * numpy.exp(1j*(g21-g11)*times)
    ca1_a1_ca1_a2 = ca1_a2 + alpha ** 2 / 2 * ca1_a2 * numpy.exp(1j*(g11-g21)*times)
    ca2_a2_ca2_a1 = ca2_a1 + alpha ** 2 / 2 * ca2_a1 * numpy.exp(1j*(g22-g12)*times)

    ca1_a2_ca1_a1 = alpha ** 2 / 2 * ca1_a2 * numpy.exp(1j*(g11-g21)*times)
    ca2_a1_ca2_a2 = alpha ** 2 / 2 * ca2_a1 * numpy.exp(1j*(g22-g12)*times)
    ca1_a1_ca2_a1 = alpha ** 2 / 2 * ca2_a1 * numpy.exp(1j*(g21-g11)*times)
    ca2_a2_ca1_a2 = alpha ** 2 / 2 * ca1_a2 * numpy.exp(1j*(g12-g22)*times)

    ca2_a1_ca1_a2 = alpha ** 4 / 4 + alpha ** 2 / 2
    ca1_a2_ca2_a1 = alpha ** 4 / 4 + alpha ** 2 / 2

    ca2_a2_ca2_a2 = alpha ** 2 / 2 + alpha ** 4 / 4
    ca1_a1_ca1_a1 = alpha ** 2 / 2 + alpha ** 4 / 4
    ca1_a1_ca2_a2 = alpha ** 4 / 4
    ca2_a2_ca1_a1 = alpha ** 4 / 4

    delta = numpy.pi / 2 - numpy.angle(ca2_a1)
    ex = numpy.exp(1j*delta)
    ex1 = numpy.exp(-1j*delta)


    exp_Jy = (ca2_a1 * ex - ca1_a2 * ex1) / 2j
    exp_Jx = (ca2_a1 * ex + ca1_a2 * ex1) / 2
    exp_Jz = (ca2_a2 - ca1_a1) / 2

    var_Jz = (ca2_a2_ca2_a2 - ca2_a2_ca1_a1 - ca1_a1_ca2_a2
        + ca1_a1_ca1_a1 - (ca2_a2-ca1_a1) ** 2) / 4
    var_Jx = (ca2_a1_ca2_a1 * ex ** 2 + ca2_a1_ca1_a2
        + ca1_a2_ca2_a1 + ca1_a2_ca1_a2 * ex1 ** 2 - (ca2_a1 * ex+ca1_a2 * ex1) ** 2) / 4
    exp_JxJz = (ca2_a1_ca2_a2 * ex - ca2_a1_ca1_a1 * ex
        + ca1_a2_ca2_a2 * ex1 - ca1_a2_ca1_a1 * ex1) / 4
    exp_JzJx = (ca2_a2_ca2_a1 * ex + ca2_a2_ca1_a2 * ex1
        - ca1_a1_ca2_a1 * ex - ca1_a1_ca1_a2 * ex1) / 4

    var_JzJx = (exp_JzJx + exp_JxJz - 2 * exp_Jz * exp_Jx)

    angle_plus_0 = numpy.arctan(var_JzJx / (var_Jz-var_Jx)) / 2 - numpy.pi / 2
    angle_plus_pi2 = angle_plus_0 + numpy.pi / 2

    var_J_theta = (
          numpy.cos(angle_plus_0) ** 2 * var_Jz
        + numpy.sin(angle_plus_0) ** 2 * var_Jx
        + numpy.sin(angle_plus_0) * numpy.cos(angle_plus_0) * var_JzJx)
    var_J_theta_pi2 = (
          numpy.cos(angle_plus_pi2) ** 2 * var_Jz
        + numpy.sin(angle_plus_pi2) ** 2 * var_Jx
        + numpy.sin(angle_plus_pi2) * numpy.cos(angle_plus_pi2) * var_JzJx)
    n0 = numpy.abs(exp_Jy) / 2

    S_theta = var_J_theta / n0
    S_theta_pi2 = var_J_theta_pi2 / n0

    return S_theta_pi2.real


def find_characteristic_time(N, a11, a12, a22):
    """
    Returns the time of the S == 1.
    """
    get_S = lambda times: get_exact_S(times, N, a11, a12, a22)
    eps = 1e-6
    t = 1.
    t_step = 1.
    S = get_S(t)

    while abs(S - 1) > 1e-6 and t_step > 1e-6:
        t_new = t + t_step
        S_new = get_S(t_new)
        if S_new > 1:
            t_step /= 2
            continue
        S = S_new
        t = t_new
        t_step *= 2

    return t


if __name__ == '__main__':

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N = 2000
    tmax = 60
    times = numpy.linspace(0.001, tmax, 200)
    S = get_exact_S(times, N, *interaction_off)

    fig = plt.figure()
    s = fig.add_subplot(111)

    s.plot(times, S)

    s.set_xlim((0, tmax))
    s.set_ylim((0, 1.2))

    fig.savefig('exact.pdf')
