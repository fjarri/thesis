import numpy
import json
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from figures import get_path
import figures.mplhelpers as mplh


def ramsey_short(fname):
    with open(get_path(__file__, 'visibility/ramsey_experimental.json')) as f:
        vis_exp = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_gpe_no_losses_vis.json')) as f:
        gpe_nl = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_wigner_vis.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_wigner_varied_pulse_vis.json')) as f:
        wig_tech = json.load(f)

    t_exp = numpy.array(vis_exp['xarray'])
    v_exp = numpy.array(vis_exp['yarray'])
    v_exp_err = numpy.array(vis_exp['yerrors'])

    gpe_t = numpy.array(gpe['times'])
    gpe_v = numpy.array(gpe['visibility'])

    gpe_nl_t = numpy.array(gpe_nl['times'])
    gpe_nl_v = numpy.array(gpe_nl['visibility'])

    wig_t = numpy.array(wig['times'])
    wig_v = numpy.array(wig['visibility'])
    wig_v_err = numpy.array(wig['visibility_errors'])

    wig_tech_t = numpy.array(wig_tech['times'])
    wig_tech_v = numpy.array(wig_tech['est_visibility'])

    gpe_N1 = numpy.array(gpe['N1'])
    gpe_N2 = numpy.array(gpe['N2'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')
    s.errorbar(t_exp, v_exp, yerr=v_exp_err, color='k', linestyle='none',
        capsize=1.5)

    # Theoretical limit of visibility
    s.plot(wig_t, 2 * numpy.sqrt(gpe_N1 * gpe_N2) / (gpe_N1 + gpe_N2), color='grey',
        linestyle=':', dashes=mplh.dash[':'])

    # GPE
    s.plot(gpe_nl_t, gpe_nl_v, color=mplh.color.f.yellow.main,
        linestyle='-.', dashes=mplh.dash['-.'])
    s.plot(gpe_t, gpe_v, color=mplh.color.f.green.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(wig_t, wig_v, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    # Wigner + technical noise
    s.plot(wig_tech_t, wig_tech_v, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 1.3))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.plot([0.04, 0.12], [0.23, 0.23], color=mplh.color.f.green.main, linestyle=':', dashes=mplh.dash[':'])
    s.text(0.14, 0.22, 'mean-field')
    s.plot([0.04, 0.12], [0.15, 0.15], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])
    s.text(0.14, 0.14, 'Wigner')
    s.plot([0.04, 0.12], [0.07, 0.07], color=mplh.color.f.blue.main, linestyle='-', dashes=mplh.dash['-'])
    s.text(0.14, 0.06, 'Wigner + tech. noise')

    s.plot([0.6, 0.68], [0.23, 0.23], color=mplh.color.f.yellow.main, linestyle='-.', dashes=mplh.dash['-.'])
    s.text(0.7, 0.22, 'mean-field, no losses')
    s.plot([0.6, 0.68], [0.15, 0.15], color='grey', linestyle=':', dashes=mplh.dash[':'])
    s.text(0.7, 0.14, 'visibility limit')
    s.errorbar([0.64], [0.07], yerr=[0.02], color='k', linestyle='none', capsize=1.5)
    s.text(0.7, 0.06, 'experiment')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_short(fname):
    with open(get_path(__file__, 'visibility/echo_experimental.json')) as f:
        vis_exp = json.load(f)
    with open(get_path(__file__, 'visibility/echo_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_wigner_vis.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/echo_wigner_varied_pulse_vis.json')) as f:
        wig_tech = json.load(f)

    t_exp = numpy.array(vis_exp['xarray'])
    v_exp = numpy.array(vis_exp['yarray'])
    v_exp_err = numpy.array(vis_exp['yerrors'])

    gpe_t = numpy.array(gpe['times'])
    gpe_v = numpy.array(gpe['visibility'])
    wig_t = numpy.array(wig['times'])
    wig_v = numpy.array(wig['visibility'])
    wig_v_err = numpy.array(wig['visibility_errors'])

    wig_tech_t = numpy.array(wig_tech['times'])
    wig_tech_v = numpy.array(wig_tech['est_visibility'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')
    s.errorbar(t_exp, v_exp, yerr=v_exp_err, color='k', linestyle='none',
        capsize=1.5)

    # GPE
    s.plot(gpe_t, gpe_v, color=mplh.color.f.green.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(wig_t, wig_v, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    # Wigner + technical noise
    s.plot(wig_tech_t, wig_tech_v, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 1.8))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.plot([0.05, 0.16], [0.23, 0.23], color=mplh.color.f.green.main, linestyle=':', dashes=mplh.dash[':'])
    s.text(0.18, 0.22, 'mean-field')
    s.plot([0.05, 0.16], [0.15, 0.15], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])
    s.text(0.18, 0.14, 'Wigner')
    s.plot([0.05, 0.16], [0.07, 0.07], color=mplh.color.f.blue.main, linestyle='-', dashes=mplh.dash['-'])
    s.text(0.18, 0.06, 'Wigner + tech. noise')

    s.errorbar([0.88], [0.07], yerr=[0.02], color='k', linestyle='none', capsize=1.5)
    s.text(0.92, 0.06, 'experiment')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def ramsey_single_run_population(fname):
    with open(get_path(__file__, 'visibility/ramsey_wigner_vis.json')) as f:
        wig = json.load(f)

    wig_t = numpy.array(wig['times'])

    wig_N1 = numpy.array(wig['N1'])
    wig_N2 = numpy.array(wig['N2'])

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$N$')

    # Pure Wigner
    s.plot(wig_t, wig_N1, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])
    s.plot(wig_t, wig_N2, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])

    s.set_xlim((0, 1.3))
    s.set_ylim((0, 55000. / 2))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.text(0.1, 2500, 'Ramsey sequence')
    s.text(1., 20500, '$\\vert 1 \\rangle$')
    s.text(1., 6500, '$\\vert 2 \\rangle$')

    fig.text(0.01, 0.92, '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_single_run_population(fname):
    with open(get_path(__file__, 'visibility/echo_wigner_single_run.json')) as f:
        wig = json.load(f)

    wig_t = numpy.array(wig['times'])

    wig_N1 = numpy.array(wig['N1'])
    wig_N2 = numpy.array(wig['N2'])

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$N$')

    # Pure Wigner
    s.plot(wig_t, wig_N1, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])
    s.plot(wig_t, wig_N2, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])

    s.set_xlim((0, 1.3))
    s.set_ylim((0, 55000. / 2))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.text(0.1, 2500, 'spin echo sequence')
    s.text(1., 4000, '$\\vert 1 \\rangle$')
    s.text(1., 13000, '$\\vert 2 \\rangle$')

    fig.text(0.01, 0.92, '(b)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def ramsey_long(fname):
    with open(get_path(__file__, 'visibility/ramsey_long_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_long_wigner_vis.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_long_wigner_varied_pulse_vis.json')) as f:
        wig_tech = json.load(f)

    t_gpe = numpy.array(gpe['times'])
    v_gpe = numpy.array(gpe['visibility'])
    t_wig = numpy.array(wig['times'])
    v_wig = numpy.array(wig['visibility'])
    v_wig_err = numpy.array(wig['visibility_errors'])
    t_tech_wig = numpy.array(wig_tech['times'])
    v_tech_wig = numpy.array(wig_tech['est_visibility'])

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')

    # GPE
    s.plot(t_gpe, v_gpe, color=mplh.color.f.green.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(t_wig, v_wig, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    s.plot(t_tech_wig, v_tech_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 3.))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.text(0.2, 0.1, 'Ramsey sequence')

    fig.text(0.01, 0.92, '(a)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_long(fname):
    with open(get_path(__file__, 'visibility/echo_long_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_long_wigner_vis.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/echo_long_wigner_varied_pulse_vis.json')) as f:
        wig_tech = json.load(f)

    t_gpe = numpy.array(gpe['times'])
    v_gpe = numpy.array(gpe['visibility'])
    t_wig = numpy.array(wig['times'])
    v_wig = numpy.array(wig['visibility'])
    v_wig_err = numpy.array(wig['visibility_errors'])
    t_tech_wig = numpy.array(wig_tech['times'])
    v_tech_wig = numpy.array(wig_tech['est_visibility'])

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{V}$')

    # GPE
    s.plot(t_gpe, v_gpe, color=mplh.color.f.green.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(t_wig, v_wig, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    s.plot(t_tech_wig, v_tech_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 3.))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.text(0.2, 0.1, 'spin echo sequence')

    fig.text(0.01, 0.92, '(b)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def combine_noises(x_final, *pairs):
    interps = [
        interp1d(x, y, kind="cubic", bounds_error=False)(x_final) for x, y in pairs]

    res = interps[0] ** 2
    for i in interps[1:]:
        res += i ** 2

    return numpy.sqrt(res)


def ramsey_noise(fname):
    with open(get_path(__file__, 'noise/ramsey_phnoise_exp.json')) as f:
        exp = json.load(f)
    with open(get_path(__file__, 'noise/ramsey_wigner_noise.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_wigner_varied_pulse_vis.json')) as f:
        tech_wig = json.load(f)

    t_exp = numpy.array(exp['xarray'])
    ph_exp = numpy.array(exp['yarray'])
    ph_exp_errors = numpy.array(exp['yerrors'])

    t_wig = numpy.array(wig['times'])
    ph_wig = numpy.array(wig['phnoise'])

    t_tech_wig = numpy.array(tech_wig['times'])
    ph_tech_wig = numpy.array(tech_wig['est_phnoises'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{\\sigma}$ (rad)')

    s.errorbar(t_exp, ph_exp, yerr=ph_exp_errors, color='k', linestyle='none',
        capsize=1.5)
    s.plot(t_wig, ph_wig, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    s.plot(t_tech_wig, ph_tech_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 0.9))
    s.set_ylim((0, 0.55))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.plot([0.02, 0.08], [0.48, 0.48], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])
    s.text(0.1, 0.47, 'Wigner')
    s.plot([0.02, 0.08], [0.44, 0.44], color=mplh.color.f.blue.main, linestyle='-', dashes=mplh.dash['-'])
    s.text(0.1, 0.43, 'Wigner + tech. noise')
    s.errorbar([0.05], [0.4], yerr=[0.01], color='k', linestyle='none', capsize=1.5)
    s.text(0.1, 0.39, 'experiment')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_noise(fname):
    with open(get_path(__file__, 'noise/echo_phnoise_exp.json')) as f:
        exp = json.load(f)
    with open(get_path(__file__, 'noise/echo_wigner_noise.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'visibility/echo_wigner_varied_pulse_vis.json')) as f:
        tech_wig = json.load(f)

    t_wig = numpy.array(wig['times'])
    ph_wig = numpy.array(wig['phnoise'])

    t_tech_wig = numpy.array(tech_wig['times'])
    ph_tech_wig = numpy.array(tech_wig['est_phnoises'])

    t_exp = numpy.array(exp['xarray'])
    ph_exp = numpy.array(exp['yarray'])
    ph_exp_errors = numpy.array(exp['yerrors'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$t$ (s)',
        ylabel='$\\mathcal{\\sigma}$ (rad)')

    s.errorbar(t_exp, ph_exp, yerr=ph_exp_errors, color='k', linestyle='none',
        capsize=1.5)
    s.plot(t_wig, ph_wig, color=mplh.color.f.red.main,
        linestyle='--', dashes=mplh.dash['--'])
    s.plot(t_tech_wig, ph_tech_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 1.6))
    s.set_ylim((0, 0.5))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.plot([0.03, 0.15], [0.43, 0.43], color=mplh.color.f.red.main, linestyle='--', dashes=mplh.dash['--'])
    s.text(0.18, 0.42, 'Wigner')
    s.plot([0.03, 0.15], [0.39, 0.39], color=mplh.color.f.blue.main, linestyle='-', dashes=mplh.dash['-'])
    s.text(0.18, 0.38, 'Wigner + tech. noise')
    s.errorbar([0.09], [0.35], yerr=[0.01], color='k', linestyle='none', capsize=1.5)
    s.text(0.18, 0.34, 'experiment')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def illustration_noise(fname, t_ms):
    with open(get_path(__file__, 'noise/ramsey_sim_phnoise_' + str(t_ms) + 'ms.json')) as f:
        meas = json.load(f)

    Pz = numpy.array(meas['Pz'])
    phis = numpy.array(meas['phis'])
    est_phase = numpy.array(meas['est_phase'])
    est_phnoise = numpy.array(meas['est_phnoise'])
    est_amp = numpy.array(meas['est_amp'])

    fig = mplh.figure(width=0.5)
    s = fig.add_subplot(111,
        xlabel='$\\phi$ (rad)',
        ylabel='$P_z$')

    phis_fit = numpy.linspace(0, numpy.pi * 2, 200)
    Pz_fit = est_amp * numpy.cos(phis_fit + est_phase)

    s.plot(phis_fit, Pz_fit, color=mplh.color.f.red.main,
        linestyle='-', dashes=mplh.dash['-'])
    s.scatter(phis, Pz, color='grey', s=1)

    s.text(numpy.pi / 2 - est_phase - est_phnoise - 0.5, 0.05, "$\\sigma")
    arrow_kwds = dict(
        shape="full",
        overhang=0, head_starts_at_zero=False, fill=False,
        length_includes_head=True)
    s.arrow(
        numpy.pi / 2 - est_phase - est_phnoise - 0.5, 0.0,
        0.45, 0.0,
        **arrow_kwds)
    s.arrow(
        numpy.pi / 2 - est_phase + est_phnoise + 0.5, 0.0,
        -0.45, 0.0,
        **arrow_kwds)

    s.set_xlim((0, 2 * numpy.pi))
    s.set_ylim((-1, 1))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    s.text(0.15, -0.85, '$t=' + str(t_ms) + '\\,\\mathrm{ms}$')

    fig.text(0.01, 0.92, '(a)' if t_ms == 20 else '(b)', fontweight='bold')

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)

