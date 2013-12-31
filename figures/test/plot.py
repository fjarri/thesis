import numpy
import json
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from figures import get_path
import figures.mplhelpers as mplh


def grid_check(fname):

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')

    for data, ls, color in [
            ('ramsey_wigner_vis.json', '-', mplh.color.f.blue.main),
            ('ramsey_wigner_test_axial2_box1_vis.json', '--', mplh.color.f.red.main),
            ('ramsey_wigner_test_axial2_box2_vis.json', ':', mplh.color.f.red.main),
            ('ramsey_wigner_test_radial2_box1_vis.json', '--', mplh.color.f.green.main),
            ('ramsey_wigner_test_radial2_box2_vis.json', ':', mplh.color.f.green.main),

            #('ramsey_gpe_vis.json', '-', mplh.color.f.yellow.main),
            #('ramsey_gpe_test_axial2_box1_vis.json', '--', mplh.color.f.red.main),
            #('ramsey_wigner_test_axial2_box2_vis.json', ':', mplh.color.f.red.main),
            #('ramsey_wigner_test_radial2_box1_vis.json', '--', mplh.color.f.green.main),
            #('ramsey_wigner_test_radial2_box2_vis.json', ':', mplh.color.f.green.main),
            ]:

        with open(get_path(__file__, data)) as f:
            wig = json.load(f)

            wig_t = numpy.array(wig['times'])
            wig_v = numpy.array(wig['visibility'])
            #wig_v_err = numpy.array(wig['visibility_errors'])

            s.plot(wig_t, wig_v, color=color, dashes=mplh.dash[ls])
            #s.plot(wig_t, wig_v + wig_v_err, color='grey', dashes=mplh.dash[ls])

    s.set_xlim((0, 1.3))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def grid_check2(fname):

    tests = [

        '_axial2_box1',
        '_axial2_box2',

        '_axial1_box1.5',
        '_axial2_box1.5',
        '_axial1_box1.1',
        '_axial2_box1.9',
        '_axial1_box1.2',
        '_axial2_box1.8',

        '_radial2_box1',
        '_radial2_box2',

        '_radial1_box1.5',
        '_radial2_box1.5',
        '_radial1_box1.1',
        '_radial2_box1.9',
        '_radial1_box1.2',
        '_radial2_box1.8',
    ]

    results = {}
    with open(get_path(__file__, 'ramsey_wigner_test_vis.json')) as f:
        data = json.load(f)
    ref_axial_spacing = data['box'][2] / data['shape'][2]
    ref_radial_spacing = data['box'][0] / data['shape'][0]
    ref_V = numpy.array(data['visibility'])
    ref_V_norm = numpy.linalg.norm(ref_V)
    ref_V_error = data['convergence']['V']
    print(ref_V_error)

    for suffix in tests:
        fn = 'ramsey_wigner_test' + suffix + '_vis.json'
        with open(get_path(__file__, fn)) as f:
            data = json.load(f)
        axial_spacing = data['box'][2] / data['shape'][2]
        radial_spacing = data['box'][0] / data['shape'][0]
        V = numpy.array(data['visibility'])

        idx = (axial_spacing / ref_axial_spacing, radial_spacing / ref_radial_spacing)
        results[idx] = (
            #numpy.linalg.norm(V - ref_V) / ref_V_norm,
            (V[-1] - ref_V[-1]) / ref_V[-1],
            data['convergence']['V'])
        print(idx, data['shape'], data['box'], results[idx])


    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='relative spacing',
        ylabel='visibility difference')

    spacings = [1.0]
    diffs = [0.0]
    errors = [ref_V_error]
    for spacing, res in results.items():
        axial, radial = spacing
        diff, error = res
        if axial == 1.:
            spacings.append(radial)
            diffs.append(diff)
            errors.append(error)
    s.errorbar(spacings, diffs, yerr=errors, linestyle='none', color='blue', label='radial')
    s.scatter(spacings, diffs, color='blue', s=10)

    spacings = [1.0]
    diffs = [0.0]
    errors = [ref_V_error]
    for spacing, res in results.items():
        axial, radial = spacing
        diff, error = res
        if radial == 1.:
            spacings.append(axial)
            diffs.append(diff)
            errors.append(error)
    s.errorbar(spacings, diffs, yerr=errors, linestyle='none', color='red', label='axial')
    s.scatter(spacings, diffs, color='red', s=10)

    s.set_xlim((0.4, 1.6))
    s.set_ylim((-0.12, 0.12))
    s.legend(loc='upper center')

    #s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def noise_check(fname):

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')

    for data, ls, color in [
            #('ramsey_wigner_test_linear_loss_vis.json', '-', mplh.color.f.blue.main),
            #('ramsey_wigner_test_linear_loss_no_K_vis.json', '--', mplh.color.f.blue.main),
            #('ramsey_wigner_test_linear_loss_no_g_vis.json', '-.', mplh.color.f.blue.main),
            #('ramsey_wigner_test_linear_loss_no_g_no_K_vis.json', ':', mplh.color.f.blue.main),
            ('ramsey_wigner_test_linear_loss_no_g_no_K_no_V_vis.json', ':', mplh.color.f.red.main),
            #('ramsey_gpe_test_linear_loss_vis.json', '-.', mplh.color.f.green.main),
            #('ramsey_wigner_test_linear_loss_zero_N_vis.json', ':', mplh.color.f.red.main),
            ]:

        with open(get_path(__file__, data)) as f:
            wig = json.load(f)

            wig_t = numpy.array(wig['times'])
            wig_N1 = numpy.array(wig['N1'])
            wig_N2 = numpy.array(wig['N2'])

            print wig_N1 + wig_N2

            s.plot(wig_t, wig_N1 + wig_N2, color=color, dashes=mplh.dash[ls])

    s.plot(wig_t, (55000 + 4096) * numpy.exp(-wig_t) - 4096, color='grey', dashes=mplh.dash[':'])

    s.set_xlim((0, 1.3))
    s.set_ylim((-1000, 55000))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def convergence(wigner, label, fname):
    with open(get_path(__file__, 'convergence_' + ('wigner' if wigner else 'gpe') + '.json')) as f:
        results = json.load(f)

    colors = {
        "RK46NL": mplh.color.f.blue,
        "RK4IP": mplh.color.f.red,
        "CDIP": mplh.color.f.green,
        "CD": mplh.color.f.yellow}
    letters = {
        "RK46NL": "(a)",
        "RK4IP": "(b)",
        "CDIP": "(d)",
        "CD": "(c)"}

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111, xlabel='steps', ylabel='errors')
    s.set_xscale('log', basex=10)
    s.set_yscale('log', basey=10)

    s.set_xlim(10**3, 10**6)
    s.set_ylim(10**(-12), 10)
    s.set_yticks([10**(-i) for i in (0, 2, 4, 6, 8, 10, 12)])

    step_nums = []
    strong_errors = []
    N_errors = []
    N_diffs = []
    SZ2_errors = []

    int_steps = [int(steps) for steps in results[label]]

    for steps in sorted(int_steps):
        result = results[label][str(steps)]
        step_nums.append(int(steps))
        strong_errors.append(result['errors']['psi'])
        N_errors.append(result['errors']['N'])
        SZ2_errors.append(result['errors']['SZ2'])
        N_diffs.append(result['N_diff'])

    s.plot(step_nums, strong_errors, label=label + ", strong errors",
        color=colors[label].main, dashes=mplh.dash['-'])
    s.plot(step_nums, N_errors, label=label + ", N errors",
        color=colors[label].main, dashes=mplh.dash['--'])
    s.plot(step_nums, SZ2_errors, label=label + ", Sz^2 errors",
        color=colors[label].main, dashes=mplh.dash[':'])

    step_nums = numpy.array(step_nums)
    s.plot(step_nums, 1e3 * (1. / step_nums) ** 1, color='grey',
        dashes=mplh.dash['-.'], linewidth=0.5)
    s.plot(step_nums, 1e6 * (1. / step_nums) ** 2, color='grey',
        dashes=mplh.dash['-.'], linewidth=0.5)
    s.plot(step_nums, 1e12 * (1. / step_nums) ** 4, color='grey',
        dashes=mplh.dash['-.'], linewidth=0.5)


    s.text(1.3*10**5, 2*10**(-1), "\\abbrev{" + label + "}")
    s.plot([1.2*10**3, 2.4*10**3], [3*10**(-7), 3*10**(-7)],
        dashes=mplh.dash['-'], color=colors[label].main)
    s.text(3*10**3, 10**(-7), '$E_{\\mathbf{\\Psi}}$')
    s.plot([1.2*10**3, 2.4*10**3], [3*10**(-9), 3*10**(-9)],
        dashes=mplh.dash['--'], color=colors[label].main)
    s.text(3*10**3, 10**(-9), '$E_2$')
    s.plot([1.2*10**3, 2.4*10**3], [3*10**(-11), 3*10**(-11)],
        dashes=mplh.dash[':'], color=colors[label].main)
    s.text(3*10**3, 10**(-11), '$E_4$')
    fig.text(0.01, 0.92, letters[label], fontweight='bold')


    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def convergence_by_time(fname):
    with open(get_path(__file__, 'convergence_wigner.json')) as f:
        results = json.load(f)

    colors = {
        "RK46NL": mplh.color.f.blue,
        "RK4IP": mplh.color.f.red,
        "CDIP": mplh.color.f.green,
        "CD": mplh.color.f.yellow}
    linestyles = {
        "RK46NL": '-',
        "RK4IP": '--',
        "CDIP": ':',
        "CD": '-.'}

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111, xlabel='integration time (s)', ylabel='$E_2$')
    s.set_xscale('log', basex=10)
    s.set_yscale('log', basey=10)

    tmin = 10
    tmax = 10 ** 4

    s.set_xlim(tmin, tmax)
    s.set_ylim(10**(-7), 1)

    #s.plot([tmin, tmax], [10**(-4), 10**(-4)], linewidth=0.5, color='grey')
    #s.plot([tmin, tmax], [10**(-5), 10**(-5)], linewidth=0.5, color='grey')
    s.fill_between([tmin, tmax], [10**(-5), 10**(-5)], [10**(-4), 10**(-4)],
        facecolor=(0.9, 0.9, 0.9), color=(0.7,0.7,0.7), linewidth=0.5)

    for label in colors:
        times = []
        N_errors = []

        int_steps = [int(steps) for steps in results[label]]

        for steps in sorted(int_steps):
            result = results[label][str(steps)]
            times.append(result['t_integration'])
            N_errors.append(result['errors']['N'])

        s.plot(times, N_errors, label=label + ", N errors",
            color=colors[label].main, dashes=mplh.dash[linestyles[label]])


    s.text(5*10**2, 10**(-2), '\\abbrev{CD}')
    s.text(2.5*10**2, 10**(-3), '\\abbrev{CDIP}')
    s.text(7.5*10**1, 10**(-2), '\\abbrev{RK4IP}')
    s.text(3*10**1, 10**(-3), '\\abbrev{RK46NL}')

    s.text(1.5*10**1, 3*10**(-6), 'target error')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)
